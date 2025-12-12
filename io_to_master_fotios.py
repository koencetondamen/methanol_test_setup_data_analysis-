import struct
import binascii
import requests
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import csv
import os


class IoLinkMaster:
    """
    SD6500 IO-Link helper:
      - talks to the AL1322 IO-Link master IoT Core
      - reads cyclic process data (PDin) from one SD6500
      - decodes totaliser, flow, pressure, temperature
      - logs samples to CSV

    CSV column order follows the process data order we use:
      totaliser -> flow -> pressure -> temperature
    """

    def __init__(self,
                 host: str,
                 auth_b64: Optional[str] = None):
        self.base = f"http://{host}"
        self.auth_b64 = auth_b64

    # -------------------------------------------------
    # Low-level helpers
    # -------------------------------------------------

    @staticmethod
    def _hex_to_bytes(s: str) -> bytes:
        """Turn a hex string (optionally with '0x') into raw bytes."""
        s = s.replace(" ", "").strip()
        if s.lower().startswith("0x"):
            s = s[2:]
        return binascii.unhexlify(s)

    def _build_iot_request(self, adr: str) -> Dict[str, Any]:
        """
        Build request payload for IO-Link master IoT Core (/getdata).
        """
        payload: Dict[str, Any] = {
            "code": "request",
            "cid": 4711,
            "adr": adr,
        }
        if self.auth_b64 is not None:
            payload["auth"] = self.auth_b64
        return payload

    def get_pdin_hex(self, port: int, timeout: float = 1.0) -> str:
        adr = f"/iolinkmaster/port[{port}]/iolinkdevice/pdin/getdata"
        payload = self._build_iot_request(adr)
        url = f"{self.base}/getdata"

        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()

        try:
            return j["data"]["value"]
        except KeyError:
            raise RuntimeError(f"Unexpected PDin response structure: {j}")

    # Convert 4-20mA to pressure
    @staticmethod
    def current_to_pressure_bar(current_mA: float,
                                span_bar: float = 100.0, # change max (m)bar of sensor
                                offset_mA: float = 4.0,
                                fullscale_mA: float = 20.0) -> float:
        
        if current_mA is None:
            return None
        
        print("current_mA: ", current_mA)

        if current_mA <= offset_mA:
            return 0.0
        if current_mA >= fullscale_mA:
            return span_bar

        fraction = (current_mA - offset_mA) / (fullscale_mA - offset_mA)
        return fraction * span_bar

    # -------------------------------------------------
    # SD6500 decoding
    # -------------------------------------------------

    def decode_sd6500_pdin(self, hex_value: str) -> Dict[str, float]:
        """
        Decode SD6500 process data (Index 40, 128 bit).

          Bytes 0..3 : totaliser_raw (Float32 BE)-> Totaliser [m³]   = totaliser_raw
          Bytes 4..5 : flow_raw      (Int16  BE) -> Flow [m³/h]      = flow_raw * 0.01
          Bytes 12..13 : pres_raw16    (Int16  BE) -> Pressure [bar]   = pres_raw16 * 0.01
          Bytes 8..9 : temp_raw16    (Int16  BE) -> Temp [°C]       = temp_raw16 * 0.01
        """
        b = self._hex_to_bytes(hex_value)

        if len(b) < 14:
            raise ValueError(f"PDin too short for SD6500: {len(b)} bytes (expected >= 14)")

        # Unpack each field separately (big-endian)
        sd65_totaliser_raw = struct.unpack(">f", b[0:4])[0]       # Float32 BE
        sd65_flow_raw      = struct.unpack(">h", b[4:6])[0]       # Int16   BE
        sd65_pres_raw16    = struct.unpack(">h", b[12:14])[0]     # Int16   BE
        sd65_temp_raw16    = struct.unpack(">h", b[8:10])[0]      # Int16   BE

        sd65_totaliser_m3  = float(sd65_totaliser_raw)            # m³
        sd65_flow_m3_h     = float(sd65_flow_raw) * 0.01          # m³/h
        sd65_pressure_bar  = float(sd65_pres_raw16) * 0.01        # bar
        sd65_temperature_c = float(sd65_temp_raw16) * 0.01        # °C

        return {
            "sd65_totaliser_m3":  sd65_totaliser_m3,
            "sd65_flow_m3_h":     sd65_flow_m3_h,
            "sd65_pressure_bar":  sd65_pressure_bar,
            "sd65_temperature_c": sd65_temperature_c,
        }

    # -------------------------------------------------
    # SD8500 decoding
    # -------------------------------------------------

    def decode_sd8500_pdin(self, hex_value: str) -> Dict[str, float]:
        """
        Decode SD8500 process data (Index 40, 128 bit).

          Bytes 0..3 : totaliser_raw (Float32 BE)-> Totaliser [m³]   = totaliser_raw
          Bytes 4..5 : flow_raw      (Int16  BE) -> Flow [m³/h]      = flow_raw * 0.01
          Bytes 12..13 : pres_raw16    (Int16  BE) -> Pressure [bar]   = pres_raw16 * 0.01
          Bytes 8..9 : temp_raw16    (Int16  BE) -> Temp [°C]       = temp_raw16 * 0.01
        """
        b = self._hex_to_bytes(hex_value)

        if len(b) < 14:
            raise ValueError(f"PDin too short for SD8500: {len(b)} bytes (expected >= 14)")

        # Unpack each field separately (big-endian)
        sd85_totaliser_raw = struct.unpack(">f", b[0:4])[0]   # Float32 BE
        sd85_flow_raw      = struct.unpack(">h", b[4:6])[0]   # Int16   BE
        sd85_pres_raw16    = struct.unpack(">h", b[12:14])[0]   # Int16   BE
        sd85_temp_raw16    = struct.unpack(">h", b[8:10])[0]  # Int16   BE

        sd85_totaliser_m3  = float(sd85_totaliser_raw)            # m³
        sd85_flow_m3_h     = float(sd85_flow_raw) * 0.01          # m³/h
        sd85_pressure_bar  = float(sd85_pres_raw16) * 0.01        # bar
        sd85_temperature_c = float(sd85_temp_raw16) * 0.01        # °C

        return {
            "sd85_totaliser_m3":  sd85_totaliser_m3,
            "sd85_flow_m3_h":     sd85_flow_m3_h,
            "sd85_pressure_bar":  sd85_pressure_bar,
            "sd85_temperature_c": sd85_temperature_c,
        }


    # -------------------------------------------------
    # AL2284 decoding
    # -------------------------------------------------

    def decode_al2284_pdin(self, hex_value: str) -> Dict[str, float]:
        """
        Decode AL2284 process data (Smart Sensor SSP 4.3.4, 4-channel float).

        Layout (cyclic input PD, 18 bytes as per SSP 4.3.4):
          Bytes 0..3   : T1 [°C]  (Float32, big-endian)
          Bytes 4..7   : T2 [°C]  (Float32, big-endian)
          Bytes 8..11  : T3 [°C]  (Float32, big-endian)
          Bytes 12..15 : T4 [°C]  (Float32, big-endian)
          Remaining bytes: status / flags (ignored here)
        """
        b = self._hex_to_bytes(hex_value)

        if len(b) < 16:
            raise ValueError(
                f"PDin too short for AL2284: {len(b)} bytes (expected >= 16)"
            )

        t1 = struct.unpack(">f", b[0:4])[0]
        t2 = struct.unpack(">f", b[4:8])[0]
        t3 = struct.unpack(">f", b[8:12])[0]
        t4 = struct.unpack(">f", b[12:16])[0]

        return {
            "al2284_t1_c": float(t1),
            "al2284_t2_c": float(t2),
            "al2284_t3_c": float(t3),
            "al2284_t4_c": float(t4),
        }

    # -------------------------------------------------
    # DP2200 decoding (4…20 mA converter)
    # -------------------------------------------------

    def decode_dp2200_pdin(self, hex_value: str) -> Dict[str, float]:
        """
        Decode DP2200 process data (Index 40, 32 bit).

        IODD (DP2200 / DP4200):
          - Process data input: RecordT (32 bit)
              * Current : IntegerT (16 bit), Big-endian
                  Range: [mA] (3600 .. 21000) * 0.001
                  32764 = NoData
                  -32760 = UL (underload)
                  32760 = OL (overload)
              * OUT1   : BooleanT (1 bit, within next byte)

        IoT Core returns 4 bytes (8 hex chars).
        """

        print("hex_value:", hex_value)

        b = self._hex_to_bytes(hex_value)

        if len(b) < 2:
            raise ValueError(
                f"PDin too short for DP2200: {len(b)} bytes (expected >= 2)"
            )

        # 16-bit signed value, Big-endian
        raw_current = struct.unpack(">h", b[0:2])[0]

        print("raw_current:", raw_current)

        # Handle special codes
        if raw_current in (32764, -32760, 32760):
            current_mA = None
        else:
            current_mA = float(raw_current) * 0.001  # µA → mA

        # OUT1 is the boolean after the current (bit 0 of next byte)
        out1_state = None
        if len(b) >= 3:
            out1_state = bool(b[2] & 0x01)

        return {
            "dp2200_current_mA": current_mA,
            "dp2200_out1_active": out1_state,
        }


    # -------------------------------------------------
    # Sampling all sensors
    # -------------------------------------------------

    def sample_all_sensors(
        self,
        sd6500_port: Optional[int]      = None,
        sd6500_2_port: Optional[int]    = None,
        sd8500_port: Optional[int]      = None,
        temp_module_port: Optional[int] = None,
        dp2200_port: Optional[int]      = None,
        timeout: float                  = 1.0,
    ) -> Dict[str, Any]:

        row: Dict[str, Any] = {}
        row["timestamp"] = datetime.now(timezone.utc).isoformat()

        # SD6500
        if sd6500_port is not None:
            try:
                sd65_hex = self.get_pdin_hex(sd6500_port, timeout=timeout)
                sd65_dec = self.decode_sd6500_pdin(sd65_hex)

                row["sd65_totaliser_m3"]   = sd65_dec["sd65_totaliser_m3"]
                row["sd65_flow_m3_h"]      = sd65_dec["sd65_flow_m3_h"]
                row["sd65_pressure_bar"]   = sd65_dec["sd65_pressure_bar"] 
                row["sd65_temperature_c"]  = sd65_dec["sd65_temperature_c"]
            except Exception:
                row["sd65_totaliser_m3"]   = None
                row["sd65_flow_m3_h"]      = None
                row["sd65_pressure_bar"]   = None
                row["sd65_temperature_c"]  = None

        # 2nd SD6500
        if sd6500_2_port is not None:
            try:
                sd65_2_hex = self.get_pdin_hex(sd6500_2_port, timeout=timeout)
                sd65_2_dec = self.decode_sd6500_pdin(sd65_2_hex)

                row["sd65_2_totaliser_m3"]   = sd65_2_dec["sd65_2_totaliser_m3"]
                row["sd65_2_flow_m3_h"]      = sd65_2_dec["sd65_2_flow_m3_h"]
                row["sd65_2_pressure_bar"]   = sd65_2_dec["sd65_2_pressure_bar"] 
                row["sd65_2_temperature_c"]  = sd65_2_dec["sd65_2_temperature_c"]
            except Exception:
                row["sd65_2_totaliser_m3"]   = None
                row["sd65_2_flow_m3_h"]      = None
                row["sd65_2_pressure_bar"]   = None
                row["sd65_2_temperature_c"]  = None

        # SD8500
        if sd8500_port is not None:
            try:
                sd85_hex = self.get_pdin_hex(sd8500_port, timeout=timeout)
                sd85_dec = self.decode_sd8500_pdin(sd85_hex)
                row["sd85_totaliser_m3"]    = sd85_dec["sd85_totaliser_m3"]
                row["sd85_flow_m3_h"]       = sd85_dec["sd85_flow_m3_h"]
                row["sd85_pressure_bar"]    = sd85_dec["sd85_pressure_bar"]
                row["sd85_temperature_c"]   = sd85_dec["sd85_temperature_c"]
            except Exception:
                row["sd85_totaliser_m3"]    = None
                row["sd85_flow_m3_h"]       = None
                row["sd85_pressure_bar"]    = None
                row["sd85_temperature_c"]   = None

        # AL2284 temp module
        if temp_module_port is not None:
            try:
                al2284_hex = self.get_pdin_hex(temp_module_port, timeout=timeout)
                al2284_dec = self.decode_al2284_pdin(al2284_hex)

                row["al2284_t1_c"] = al2284_dec["al2284_t1_c"]
                row["al2284_t2_c"] = al2284_dec["al2284_t2_c"]
                row["al2284_t3_c"] = al2284_dec["al2284_t3_c"]
                row["al2284_t4_c"] = al2284_dec["al2284_t4_c"]
            except Exception:
                row["al2284_t1_c"] = None
                row["al2284_t2_c"] = None
                row["al2284_t3_c"] = None
                row["al2284_t4_c"] = None

        # DP2200 4…20 mA converter
        if dp2200_port is not None:
            try:
                dp_hex = self.get_pdin_hex(dp2200_port, timeout=timeout)
                dp_dec = self.decode_dp2200_pdin(dp_hex)

                row["dp2200_current_mA"]   = dp_dec["dp2200_current_mA"]
                row["dp2200_pressure_bar"] = self.current_to_pressure_bar(
                    dp_dec["dp2200_current_mA"],
                    span_bar=100.0,  # adjust to your sensor range
                )
            except Exception:
                row["dp2200_current_mA"]   = None
                row["dp2200_pressure_bar"] = None
                
        return row


    # -------------------------------------------------
    # Logging → CSV
    # -------------------------------------------------

    def log_loop_to_csv(
        self,
        csv_path: str,
        period_s: float = 1.0,
        sensors_to_log: Optional[list] = None,
        sd6500_port: Optional[int] = None,
        sd6500_2_port: Optional[int] = None,
        sd8500_port: Optional[int] = None,
        temp_module_port: Optional[int] = None,
        dp2200_port: Optional[int] = None,
        timeout: float = 1.0,
    ) -> None:
        """
        Periodically sample selected sensors and append to csv_path.

        Runs until KeyboardInterrupt.
        """

        # build CSV header based on sensors_to_log
        fieldnames = ["timestamp"]

        if "sd6500" in sensors_to_log:
            fieldnames += ["sd65_totaliser_m3", "sd65_flow_m3_h", "sd65_pressure_bar", "sd65_temperature_c"]
        if "sd6500_2" in sensors_to_log:
            fieldnames += ["sd65_2_totaliser_m3", "sd65_2_flow_m3_h", "sd65_2_pressure_bar", "sd65_2_temperature_c"]
        if "sd8500" in sensors_to_log:
            fieldnames += ["sd85_totaliser_m3", "sd85_flow_m3_h", "sd85_pressure_bar", "sd85_temperature_c"]
        if "al2284" in sensors_to_log:
            fieldnames += ["al2284_t1_c", "al2284_t2_c", "al2284_t3_c", "al2284_t4_c"]
        if "dp2200" in sensors_to_log:
            fieldnames += ["dp2200_pressure_bar"]


        # make sure CSV exists + header
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print(f"Logging to {os.path.abspath(csv_path)} every {period_s}s")
        print("Fields:", fieldnames)

        try:
            while True:
                # take one snapshot
                row_full = self.sample_all_sensors(
                    sd6500_port          = sd6500_port        if "sd6500"         in sensors_to_log else None,
                    sd6500_2_port        = sd6500_2_port      if "sd6500_2"       in sensors_to_log else None,
                    sd8500_port          = sd8500_port        if "sd8500"         in sensors_to_log else None,
                    temp_module_port     = temp_module_port   if "al2284"         in sensors_to_log else None,
                    dp2200_port          = dp2200_port        if "dp2200"         in sensors_to_log else None,
                    timeout          = timeout,
                )

                # keep only the columns we advertised, and in the expected order
                row_to_write = {k: row_full.get(k, None) for k in fieldnames}

                # append to CSV
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row_to_write)

                time.sleep(period_s)

        except KeyboardInterrupt:
            print("Stopped logging.")
