# methanol_dashboard/simulation.py

"""
Simulation mode for the methanol / N2 test setup.

Provides SimulatedIoLinkMaster with the same interface as IoLinkMaster,
so AcquisitionManager and the dashboard can run without real hardware.
"""

import math
import random
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any


class SimulatedIoLinkMaster:
    """
    Drop-in replacement for IoLinkMaster used when USE_SIMULATION=True.

    The method `sample_all_sensors(...)` returns a dict with the same keys
    as the real implementation would produce:

    - timestamps
    - sd8500_* (flow, pressure, temperature, totaliser)
    - sd6500_1_* and sd6500_2_*
    - senxtx_o2_current_mA
    - dewpoint_michell_current_mA
    - dewpoint_banner_1_degC, dewpoint_banner_2_degC
    - pt100_1_degC .. pt100_4_degC
    """

    def __init__(self, host: str) -> None:
        # Keep signature compatible, but ignore host/auth.
        self._t0 = time.time()
        self._rng = random.Random(42)

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
        """Return one synthetic sample row for all sensors."""

        now_utc = datetime.now(timezone.utc)
        t_rel = time.time() - self._t0  # seconds since start

        # Helper for a smooth signal + a bit of noise
        def wave(base: float, amp: float, period_s: float, noise: float) -> float:
            val = base + amp * math.sin(2 * math.pi * t_rel / period_s)
            val += self._rng.gauss(0.0, noise)
            return val

        row: Dict[str, Any] = {
            "timestamp_utc": now_utc.isoformat(),
            "timestamp_unix_s": time.time(),
        }

        # ----------------------------
        # SD8500: flow / pressure / temp / totaliser
        # ----------------------------
        if sd8500_port is not None:
            sd8500_flow = max(0.0, wave(base=5.0, amp=1.0, period_s=60.0, noise=0.05))
            sd8500_pressure = max(0.0, wave(base=2.0, amp=0.3, period_s=90.0, noise=0.02))
            sd8500_temp = wave(base=25.0, amp=2.0, period_s=300.0, noise=0.1)
            # Just integrate flow crudely for totaliser
            totaliser = 5.0 * t_rel / 3600.0  # m³, assuming avg 5 m³/h

            row.update(
                {
                    "sd8500_flow_m3_h": sd8500_flow,
                    "sd8500_pressure_bar": sd8500_pressure,
                    "sd8500_temperature_c": sd8500_temp,
                    "sd8500_totaliser_m3": totaliser,
                }
            )

        # ----------------------------
        # SD6500 #1
        # ----------------------------
        if sd6500_1_port is not None:
            flow = max(0.0, wave(base=3.0, amp=0.7, period_s=50.0, noise=0.05))
            pressure = max(0.0, wave(base=1.5, amp=0.2, period_s=80.0, noise=0.02))
            temp = wave(base=23.0, amp=1.5, period_s=280.0, noise=0.1)
            totaliser = 3.0 * t_rel / 3600.0

            row.update(
                {
                    "sd6500_1_flow_m3_h": flow,
                    "sd6500_1_pressure_bar": pressure,
                    "sd6500_1_temperature_c": temp,
                    "sd6500_1_totaliser_m3": totaliser,
                }
            )

        # ----------------------------
        # SD6500 #2
        # ----------------------------
        if sd6500_2_port is not None:
            flow = max(0.0, wave(base=4.0, amp=0.8, period_s=70.0, noise=0.05))
            pressure = max(0.0, wave(base=1.8, amp=0.25, period_s=95.0, noise=0.02))
            temp = wave(base=24.0, amp=1.2, period_s=310.0, noise=0.1)
            totaliser = 4.0 * t_rel / 3600.0

            row.update(
                {
                    "sd6500_2_flow_m3_h": flow,
                    "sd6500_2_pressure_bar": pressure,
                    "sd6500_2_temperature_c": temp,
                    "sd6500_2_totaliser_m3": totaliser,
                }
            )

        # ----------------------------
        # SenxTx O2 via analogue current
        # ----------------------------
        if senxtx_port is not None:
            # Simulate 4–20 mA with a slow drift
            current = wave(base=12.0, amp=3.0, period_s=600.0, noise=0.05)
            current = min(max(current, 4.0), 20.0)
            row["senxtx_o2_current_mA"] = current

        # ----------------------------
        # Dewpoint Michell via analogue current
        # ----------------------------
        if michell_port is not None:
            current = wave(base=10.0, amp=4.0, period_s=400.0, noise=0.05)
            current = min(max(current, 4.0), 20.0)
            row["dewpoint_michell_current_mA"] = current

        # ----------------------------
        # Dewpoint Banner #1 and #2 via Modbus–IO–Link
        # ----------------------------
        if banner_dp1_port is not None:
            humidity = wave(base=60.0, amp=15.0, period_s=700.0, noise=1.0)
            temperature = wave(base=20.0, amp=5.0, period_s=500.0, noise=0.2)
            dewpoint = wave(base=10.0, amp=3.0, period_s=500.0, noise=0.2)

            row["dewpoint_banner_1_Humidity"] = humidity
            row["dewpoint_banner_1_degreeC"] = temperature
            row["dewpoint_banner_1_dewpoint"] = dewpoint

        if banner_dp2_port is not None:
            dew2 = wave(base=-3.0, amp=2.5, period_s=450.0, noise=0.2)
            row["dewpoint_banner_2_degC"] = dew2

        # ----------------------------
        # PT100 #1..#4 via AL2284
        # ----------------------------
        if pt100_module_port is not None:
            base_temp = wave(base=30.0, amp=1.0, period_s=900.0, noise=0.1)
            row.update(
                {
                    "pt100_1_degC": base_temp + self._rng.gauss(0.0, 0.1),
                    "pt100_2_degC": base_temp + self._rng.gauss(0.1, 0.1),
                    "pt100_3_degC": base_temp + self._rng.gauss(-0.1, 0.1),
                    "pt100_4_degC": base_temp + self._rng.gauss(0.2, 0.1),
                }
            )

        return row
