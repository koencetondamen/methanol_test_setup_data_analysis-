import struct
import binascii
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import requests


class IoLinkMaster:
    
    """
    IO-Link master helper for the methanol/N2 test setup.

    Devices:
    - SD8500 (flow, pressure, temp)
    - SD6500 #1 and #2 (flow, pressure, temp)
    - SenxTx analogue (current via IO-Link analogue module, e.g. DP2200)
    - Dewpoint Michell analogue (current via IO-Link analogue module)
    - 2x Dewpoint Banner via Modbus–IO–Link converter
    - 4x PT100 via AL2284 4-channel temperature module
    """

    def __init__(self, host: str, auth_b64: Optional[str] = None) -> None:
        self.host = host
        self.auth_b64 = auth_b64.rstrip() if auth_b64 else None

    # ------------------------------
    # Low-level IO-Link / HTTP
    # ------------------------------

    def _request_json(self, path: str, timeout: float = 1.0) -> Dict[str, Any]:

        """
        Simple GET-based access to the IoT core.

        If your master needs POST with a JSON body, you can adapt this to match
        the working version you already have.
        """

        url = f"http://{self.host}{path}"
        headers = {}
        if self.auth_b64:
            headers["Authorization"] = self.auth_b64

        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get_pdin_hex(self, port: int, timeout: float = 1.0) -> Optional[str]:
        """
        Read PDin hex value from one port.

        Returns hex string without '0x' or None if no data.
        """
        # Path may differ per master – adjust when you connect.
        data = self._request_json(
            f"/iolinkmaster/port[{port}]/iolinkdevice/pdin/getdata",
            timeout=timeout,
        )
        # Common pattern: {"data": "0xAABBCCDD"}
        hex_value = data.get("data") or data.get("value")
        if not hex_value:
            return None

        return hex_value.replace("0x", "").replace("0X", "").strip()

    @staticmethod
    def _hex_to_bytes(hex_value: str) -> bytes:
        return binascii.unhexlify(hex_value)

    # ------------------------------
    # Common helpers for SD6500/8500
    # ------------------------------

    def _decode_sd_common(self, hex_value: str) -> Dict[str, float]:
        """
        Common decoding for SD6500 / SD8500, 128-bit PD.

        Bytes 0..3  : totaliser_raw (Float32 BE) -> [m³]
        Bytes 4..5  : flow_raw      (Int16  BE)  -> [m³/h] * 0.01
        Bytes 8..9  : temp_raw16    (Int16  BE)  -> [°C]   * 0.01
        Bytes 12..13: pres_raw16    (Int16  BE)  -> [bar]  * 0.01
        """
        b = self._hex_to_bytes(hex_value)

        if len(b) < 14:
            raise ValueError(f"PDin too short for SD device: {len(b)} bytes (expected >= 14)")

        totaliser_raw = struct.unpack(">f", b[0:4])[0]
        flow_raw = struct.unpack(">h", b[4:6])[0]
        temp_raw16 = struct.unpack(">h", b[8:10])[0]
        pres_raw16 = struct.unpack(">h", b[12:14])[0]

        return {
            "totaliser_m3": float(totaliser_raw),
            "flow_m3_h": float(flow_raw) * 0.01,
            "temperature_c": float(temp_raw16) * 0.01,
            "pressure_bar": float(pres_raw16) * 0.01,
        }

    def decode_sd6500_pdin(self, hex_value: str, prefix: str) -> Dict[str, float]:
        """Decode SD6500 PDin with a name prefix (e.g. 'sd6500_1')."""
        base = self._decode_sd_common(hex_value)
        return {f"{prefix}_{k}": v for k, v in base.items()}

    def decode_sd8500_pdin(self, hex_value: str, prefix: str = "sd8500") -> Dict[str, float]:
        """Decode SD8500 PDin with a name prefix (default 'sd8500')."""
        base = self._decode_sd_common(hex_value)
        return {f"{prefix}_{k}": v for k, v in base.items()}

    # ------------------------------
    # AL2284 PT100 module
    # ------------------------------

    def decode_al2284_pdin(self, hex_value: str) -> Dict[str, float]:
        """
        Decode AL2284 process data (4-channel float, big-endian).

        Bytes 0..3   : T1 [°C] float32 BE
        Bytes 4..7   : T2 [°C]
        Bytes 8..11  : T3 [°C]
        Bytes 12..15 : T4 [°C]
        """
        b = self._hex_to_bytes(hex_value)

        if len(b) < 16:
            raise ValueError(f"PDin too short for AL2284: {len(b)} bytes (expected >= 16)")

        t1 = struct.unpack(">f", b[0:4])[0]
        t2 = struct.unpack(">f", b[4:8])[0]
        t3 = struct.unpack(">f", b[8:12])[0]
        t4 = struct.unpack(">f", b[12:16])[0]

        return {
            "pt100_1_degC": float(t1),
            "pt100_2_degC": float(t2),
            "pt100_3_degC": float(t3),
            "pt100_4_degC": float(t4),
        }

    # ------------------------------
    # Analogue IO-Link (DP2200-like)
    # ------------------------------

    def decode_analog_current_pdin(self, hex_value: str, prefix: str) -> Dict[str, float]:
        """
        Decode 16-bit current (4-20 mA style) from an analogue IO-Link module.

        According to DP2200 IODD:
        - Process data input: 16-bit signed integer, big-endian
        - Range: [mA] (3600 .. 21000) * 0.001
        - 32764 = NoData, -32760 = UL, 32760 = OL
        """
        b = self._hex_to_bytes(hex_value)
        if len(b) < 2:
            raise ValueError(f"PDin too short for analogue current: {len(b)} bytes (expected >= 2)")

        raw = struct.unpack(">h", b[0:2])[0]
        if raw in (32764, -32760, 32760):
            current_mA = float("nan")  # or None if you prefer
        else:
            current_mA = float(raw) * 0.001

        return {f"{prefix}_current_mA": current_mA}

    # ------------------------------
    # Banner dewpoint via Modbus–IO–Link
    # ------------------------------

    def decode_banner_dewpoint_pdin(self, hex_value: str, prefix: str) -> Dict[str, float]:
        """
        Decode dewpoint from a Modbus–IO–Link converter.

        This is a placeholder: adjust scaling once you have the IODD.
        For now, treat the first 16 bits as a signed integer and scale by 0.1.
        """
        b = self._hex_to_bytes(hex_value)
        if len(b) < 2:
            raise ValueError(f"PDin too short for Banner dewpoint: {len(b)} bytes (expected >= 2)")

        raw = struct.unpack(">h", b[0:2])[0]
        dewpoint_degC = float(raw) * 0.1  # TODO: verify scaling

        return {f"{prefix}_degC": dewpoint_degC}

    # ------------------------------
    # High-level sampling
    # ------------------------------

    def sample_all_sensors(
        self,
        *,
        sd8500_port: Optional[int] = None,
        sd6500_1_port: Optional[int] = None,
        sd6500_2_port: Optional[int] = None,
        senxtx_port: Optional[int] = None,
        michell_port: Optional[int] = None,
        banner_dp1_port: Optional[int] = None,
        banner_dp2_port: Optional[int] = None,
        pt100_module_port: Optional[int] = None,
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Take a snapshot of all configured sensors and return a single row dict.

        The dict always contains:
        - timestamp_utc
        - timestamp_unix_s

        Plus keys for each successfully decoded sensor.
        """
        row: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_unix_s": time.time(),
        }

        # SD8500 ------------------------------------------------
        if sd8500_port is not None:
            try:
                hx = self.get_pdin_hex(sd8500_port, timeout=timeout)
                if hx:
                    row.update(self.decode_sd8500_pdin(hx, prefix="sd8500"))
            except Exception as exc:
                print(f"[IoLinkMaster] SD8500 error: {exc}")

        # SD6500 #1 ---------------------------------------------
        if sd6500_1_port is not None:
            try:
                hx = self.get_pdin_hex(sd6500_1_port, timeout=timeout)
                if hx:
                    row.update(self.decode_sd6500_pdin(hx, prefix="sd6500_1"))
            except Exception as exc:
                print(f"[IoLinkMaster] SD6500 #1 error: {exc}")

        # SD6500 #2 ---------------------------------------------
        if sd6500_2_port is not None:
            try:
                hx = self.get_pdin_hex(sd6500_2_port, timeout=timeout)
                if hx:
                    row.update(self.decode_sd6500_pdin(hx, prefix="sd6500_2"))
            except Exception as exc:
                print(f"[IoLinkMaster] SD6500 #2 error: {exc}")

        # SenxTx analogue ---------------------------------------
        if senxtx_port is not None:
            try:
                hx = self.get_pdin_hex(senxtx_port, timeout=timeout)
                if hx:
                    row.update(self.decode_analog_current_pdin(hx, prefix="senxtx_o2"))
            except Exception as exc:
                print(f"[IoLinkMaster] SenxTx error: {exc}")

        # Dewpoint Michell analogue -----------------------------
        if michell_port is not None:
            try:
                hx = self.get_pdin_hex(michell_port, timeout=timeout)
                if hx:
                    row.update(self.decode_analog_current_pdin(hx, prefix="dewpoint_michell"))
            except Exception as exc:
                print(f"[IoLinkMaster] Michell dewpoint error: {exc}")

        # Dewpoint Banner #1 ------------------------------------
        if banner_dp1_port is not None:
            try:
                hx = self.get_pdin_hex(banner_dp1_port, timeout=timeout)
                if hx:
                    row.update(self.decode_banner_dewpoint_pdin(hx, prefix="dewpoint_banner_1"))
            except Exception as exc:
                print(f"[IoLinkMaster] Banner dewpoint #1 error: {exc}")

        # Dewpoint Banner #2 ------------------------------------
        if banner_dp2_port is not None:
            try:
                hx = self.get_pdin_hex(banner_dp2_port, timeout=timeout)
                if hx:
                    row.update(self.decode_banner_dewpoint_pdin(hx, prefix="dewpoint_banner_2"))
            except Exception as exc:
                print(f"[IoLinkMaster] Banner dewpoint #2 error: {exc}")

        # PT100 module (AL2284) --------------------------------
        if pt100_module_port is not None:
            try:
                hx = self.get_pdin_hex(pt100_module_port, timeout=timeout)
                if hx:
                    row.update(self.decode_al2284_pdin(hx))
            except Exception as exc:
                print(f"[IoLinkMaster] PT100 module error: {exc}")

        return row
