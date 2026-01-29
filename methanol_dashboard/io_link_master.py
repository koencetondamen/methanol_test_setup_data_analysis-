import struct
import binascii
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import requests
import math

class IoLinkMaster:

    """
    IO-Link master helper for the methanol/N2 test setup.
    """

    def __init__(self, host: str) -> None:
        self.host = host

    # ------------------------------
    # Low-level IO-Link / HTTP
    # ------------------------------
    
    # get JSON data from ethernet
    def _request_json(self, path: str, timeout: float = 1.0) -> Dict[str, Any]:

        """
        Simple GET-based access to the IoT core.
        """

        url = f"http://{self.host}{path}" # host : IP adress, path : defined with port 

        resp = requests.get(url, timeout=timeout) # 
        resp.raise_for_status()
        return resp.json() # decode json to python object

    # json string to hex
    def get_pdin_hex(self, port: int, timeout: float = 1.0) -> Optional[str]:

        """
        Read PDin hex value from one port.

        Returns hex string without '0x' or None if no data.
        """

        # get the data of the port
        data = self._request_json(
            f"/iolinkmaster/port[{port}]/iolinkdevice/pdin/getdata",
            timeout=timeout,
        )

        hex_value = data["data"]["value"]

        if not hex_value:
            return None
        
        return hex_value.replace("0x", "").replace("0X", "").strip()

    # heximal to bites
    @staticmethod
    def _hex_to_bytes(hex_value: str) -> bytes:
        return binascii.unhexlify(hex_value)
  
    # ------------------------------
    # Component and Module functions
    # ------------------------------

    # Sensor: decode sd6500
    def decode_sd6500_pdin(self, hex_value: str, prefix: str) -> Dict[str, float]:
        
        """
        Decode SD6500 PDin.

        Byte layout (128-bit PD):
            Bytes 0..3   : totaliser_raw (Float32 BE) -> [m³]
            Bytes 4..5   : flow_raw      (Int16  BE)  -> [m³/h] * 0.01
            Bytes 8..9   : temp_raw16    (Int16  BE)  -> [°C]   * 0.01
            Bytes 12..13 : pres_raw16    (Int16  BE)  -> [bar]  * 0.01
            Byte  15     : UIntegerT     (4 Bit)      -> 
                        0 - Ok, 
                        1 - Maintenance, 
                        2 - Out of spec, 
                        3 - Functional Check, 
                        4 - Failure
        """

        b = bytes.fromhex(hex_value)

        if len(b) < 14:
            raise ValueError(f"PDin too short for SD6500: {len(b)} bytes (expected >= 14)")

        totaliser_raw = struct.unpack(">f", b[0:4])[0]
        flow_raw      = struct.unpack(">h", b[4:6])[0]
        temp_raw16    = struct.unpack(">h", b[8:10])[0]
        pres_raw16    = struct.unpack(">h", b[12:14])[0]

        # Extract the status (lower 4 bits of byte 15)
        status_code = b[15] & 0x0F
        status_map = {
            0: "OK",
            1: "Maintenance",
            2: "Out of spec",
            3: "Functional check",
            4: "Failure"
        }

        status = status_map.get(status_code, f"Unknown ({status_code})")

        #prefix = "sd6500_"

        return {
            f"{prefix}_totaliser_m3":   float(totaliser_raw),
            f"{prefix}_flow_m3_h":      float(flow_raw) * 0.01,
            f"{prefix}_temperature_c":  float(temp_raw16) * 0.01,
            f"{prefix}_pressure_bar":   float(pres_raw16) * 0.01,
            f"{prefix}_status":         status,
        }

    # Sensor: decode sd8500
    def decode_sd8500_pdin(self, hex_value: str) -> Dict[str, float]:
        """
        Decode SD8500 PDin.

        Byte layout (128-bit PD):
            Bytes 0..3   : totaliser_raw (Float32 BE) -> [m³]
            Bytes 4..5   : flow_raw      (Int16  BE)  -> [m³/h] * 0.01
            Bytes 8..9   : temp_raw16    (Int16  BE)  -> [°C]   * 0.01
            Bytes 12..13 : pres_raw16    (Int16  BE)  -> [bar]  * 0.01
            Byte  15     : UIntegerT     (4 Bit)      -> 
                        0 - Ok, 
                        1 - Maintenance, 
                        2 - Out of spec, 
                        3 - Functional Check, 
                        4 - Failure
        """

        b = bytes.fromhex(hex_value)

        if len(b) < 16:
            raise ValueError(f"PDin too short for SD8500: {len(b)} bytes (expected >= 16)")

        totaliser_raw = struct.unpack(">f", b[0:4])[0]
        flow_raw      = struct.unpack(">h", b[4:6])[0]
        temp_raw16    = struct.unpack(">h", b[8:10])[0]
        pres_raw16    = struct.unpack(">h", b[12:14])[0]

        # Extract the status (lower 4 bits of byte 15)
        status_code = b[15] & 0x0F
        status_map = {
            0: "OK",
            1: "Maintenance",
            2: "Out of spec",
            3: "Functional check",
            4: "Failure"
        }

        status = status_map.get(status_code, f"Unknown ({status_code})")

        prefix = "sd8500_"
        return {
            f"{prefix}totaliser_m3":   float(totaliser_raw),
            f"{prefix}flow_m3_h":      float(flow_raw) * 0.01,
            f"{prefix}temperature_c":  float(temp_raw16) * 0.01,
            f"{prefix}pressure_bar":   float(pres_raw16) * 0.01,
            f"{prefix}status":         status,
        }

    # module: 4 temperature sensors 
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

        if t1 == 3.299999965482712e+38:
            t1 = float("nan")
        if t2 == 3.299999965482712e+38:
            t2 = float("nan")
        if t3 == 3.299999965482712e+38:
            t3 = float("nan")
        if t4 == 3.299999965482712e+38:
            t4 = float("nan")

        return {
           "pt100_1_degC": float(t1),
           "pt100_2_degC": float(t2),
           "pt100_3_degC": float(t3),
           "pt100_4_degC": float(t4),
        }
 
    # Sensor: Senz-Tx oxygen sensor
    def decode_senxtx_oxygen(self, hex_value: str) -> Dict[str, float]:
       
        """
        Decode SenzTx oxygen signal via DP2200 + IO-Link master.

        DP2200 process data (BigEndian):
            - 4 bytes total (RecordT 32 bit)
            - Bytes 0..1: Current, IntegerT (16 bit), scaled:
                current_mA = current_raw * 0.001
            Special values:
                32764  (0x7FFC) : NoData
                -32760 (0x8008) : Underload (UL)
                32760  (0x7FF8) : Overload (OL)
            - Remaining bits contain OUT1 status (ignored here).

        SenzTx mapping:
            4–20 mA  ->  0–25 % O₂  (linear)
        """
        b = bytes.fromhex(hex_value)
        if len(b) < 2:
            raise ValueError(f"Expected at least 2 bytes, got {len(b)} from {hex_value!r}")

        # 16-bit signed integer, big-endian
        current_raw = struct.unpack(">h", b[0:2])[0]

        # Handle DP2200 special codes → treat as NaN
        if current_raw in (32764, -32760, 32760):
            current_mA = float("nan")
        else:
            current_mA = current_raw * 0.001  # raw → mA

        # 4–20 mA → 0–25 % O₂
        if math.isnan(current_mA):
            oxygen_percent = float("nan")
        else:
            oxygen_percent = (current_mA - 4.0) / 16.0 * 25.0  # (20 - 4) = 16 mA span
            # Clamp to [0, 25]
            if oxygen_percent < 0.0:
                oxygen_percent = 0.0
            elif oxygen_percent > 25.0:
                oxygen_percent = 25.0

        return {
            "senxtx_o2_current_mA": float(current_mA),
            "senxtx_o2_oxygen_percent": float(oxygen_percent),
        }

    # Banner dewpoint via Modbus–IO–Link (S15C)
    def decode_banner_dewpoint_pdin(self, hex_value: str, prefix: str) -> Dict[str, float]:

        """
        Decode Banner dewpoint sensor values coming via a Modbus–IO–Link converter (Banner S15C).

        S24 Modbus holding register layout:
            40001 : Humidity      (%RH)  -> raw / 100   (UInt16)
            40002 : Temperature   (°C)   -> raw / 20    (Int16)
            40004 : Dew Point     (°C)   -> raw / 100   (Int16)

        Real S15C PDin (as returned by IFM) is a fixed-length byte array padded with zeros.
        The three register values show up at the END of PDin, followed by a status word:
            ... [dewpoint_raw_u16, temp_raw_u16, humidity_raw_u16, status_u16]
        Example tail: 0401 01EE 0FB2 0710
        """

        b = self._hex_to_bytes(hex_value)

        if len(b) < 8 or (len(b) % 2) != 0:
            raise ValueError(
                f"PDin invalid length: {len(b)} bytes (expected >= 8 and even)"
            )

        # Interpret the whole PDin as big-endian 16-bit words
        words = struct.unpack(f">{len(b)//2}H", b)

        # Use the last 4 words: [dewpoint, temperature, humidity, status]
        dew_u16, temp_u16, hum_u16, status_u16 = words[-4], words[-3], words[-2], words[-1]

        # Convert unsigned 16-bit to signed 16-bit for temp/dewpoint (two's complement)
        def u16_to_s16(u: int) -> int:
            return u - 0x10000 if (u & 0x8000) else u

        dew_raw = u16_to_s16(dew_u16)     # Int16
        temp_raw = u16_to_s16(temp_u16)   # Int16
        hum_raw = hum_u16                # UInt16

        humidity_rh = hum_raw / 100.0
        temperature_c = temp_raw / 20.0
        dewpoint_c = dew_raw / 100.0

        print("PDin tail words:", [f"0x{w:04X}" for w in (dew_u16, temp_u16, hum_u16, status_u16)])
        print("humidity:", humidity_rh)
        print("temperature:", temperature_c)
        print("dewpoint:", dewpoint_c)

        return {
            f"{prefix}_Humidity": float(humidity_rh),
            f"{prefix}_degreeC": float(temperature_c),
            f"{prefix}_dewpoint": float(dewpoint_c),
        }

    # Sensor: Michell dew point via DP2200
    def decode_michell_dewpoint(self, hex_value: str) -> Dict[str, float]:
        """
        Decode Michell dew point transmitter signal via DP2200 + IO-Link master.

        DP2200 process data (BigEndian):
            - 4 bytes total (RecordT 32 bit)
            - Bytes 0..1: Current, IntegerT (16 bit), scaled:
                current_mA = current_raw * 0.001
            Special values:
                32764  (0x7FFC) : NoData
                -32760 (0x8008) : Underload (UL)
                32760  (0x7FF8) : Overload (OL)
            - Remaining bits contain OUT1 status (ignored here).

        Michell dew point mapping:
            4–20 mA → -110 to +20 °C (linear) (NOTE: sticker says -100, 20)
        """

        b = bytes.fromhex(hex_value)
        if len(b) < 2:
            raise ValueError(f"Expected at least 2 bytes, got {len(b)} from {hex_value!r}")

        # 16-bit signed integer, big-endian
        current_raw = struct.unpack(">h", b[0:2])[0]

        # Handle DP2200 special codes → treat as NaN
        if current_raw in (32764, -32760, 32760):
            current_mA = float("nan")
        else:
            current_mA = current_raw * 0.001  # raw → mA

        # 4–20 mA → -110 to +20 °C
        if math.isnan(current_mA):
            dewpoint_c = float("nan")
        else:
            # Linear conversion: (value - 4 mA) / (16 mA) * range_span + min_value
            dewpoint_c = (current_mA - 4.0) / 16.0 * (20.0 - (-110.0)) + (-110.0)
            # Clamp to valid range
            dewpoint_c = max(-110.0, min(20.0, dewpoint_c))

        return {
            "michell_dewpoint_current_mA": float(current_mA),
            "michell_dewpoint_c": float(dewpoint_c),
        }

    # ------------------------------
    # Sampling
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
        banner_dp3_port: Optional[int] = None,
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
                print("SD8500 value:", hx)
                if hx:
                    row.update(self.decode_sd8500_pdin(hx))
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
                print("hx:",hx)
                if hx:
                    row.update(self.decode_senxtx_oxygen(hx))
            except Exception as exc:
                print(f"[IoLinkMaster] SenxTx error: {exc}")

        # Dewpoint Michell analogue -----------------------------
        if michell_port is not None:
            try:
                hx = self.get_pdin_hex(michell_port, timeout=timeout)
                print("hx michell dewpoint", hx)
                if hx:
                    row.update(self.decode_michell_dewpoint(hx))
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

        # Dewpoint Banner #3 ------------------------------------
        if banner_dp3_port is not None:
            try:
                hx = self.get_pdin_hex(banner_dp3_port, timeout=timeout)
                if hx:
                    row.update(self.decode_banner_dewpoint_pdin(hx, prefix="dewpoint_banner_3"))
            except Exception as exc:
                print(f"[IoLinkMaster] Banner dewpoint #3 error: {exc}")

        # PT100 module (AL2284) --------------------------------
        if pt100_module_port is not None:
            try:
                hx = self.get_pdin_hex(pt100_module_port, timeout=timeout)
                print("pt100:", hx)
                if hx:
                    row.update(self.decode_al2284_pdin(hx))
            except Exception as exc:
                print(f"[IoLinkMaster] PT100 module error: {exc}")

        print("row:", row)
        return row
