"""
Simulation mode for the methanol / N2 test setup.

Provides SimulatedIoLinkMaster with the same interface as IoLinkMaster,
so AcquisitionManager and the dashboard can run without real hardware.

This simulation is kept in sync with the *current* logged columns:

- timestamps
- sd8500_* (flow, pressure, status, temperature, totaliser)
- sd6500_1_* and sd6500_2_* (flow, pressure, status, temperature, totaliser)
- senxtx_o2_current_mA, senxtx_o2_oxygen_percent
- michell_dewpoint_current_mA, michell_dewpoint_c
- dewpoint_banner_1_Humidity, dewpoint_banner_1_degreeC, dewpoint_banner_1_dewpoint
- dewpoint_banner_2_Humidity, dewpoint_banner_2_degreeC, dewpoint_banner_2_dewpoint
- dewpoint_banner_3_Humidity, dewpoint_banner_3_degreeC, dewpoint_banner_3_dewpoint
- pt100_1_degC .. pt100_4_degC

Some backward-compatible aliases are also emitted:
- dewpoint_michell_current_mA (alias of michell_dewpoint_current_mA)
- dewpoint_banner_2_degC (alias of dewpoint_banner_2_degreeC)
"""

from __future__ import annotations

import math
import random
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


class SimulatedIoLinkMaster:
    """
    Drop-in replacement for IoLinkMaster used when USE_SIMULATION=True.

    The method `sample_all_sensors(...)` returns a dict with the same keys
    as the real implementation would produce.
    """

    def __init__(self, host: str) -> None:
        # Keep signature compatible, but ignore host/auth.
        self._t0 = time.time()
        self._rng = random.Random(42)

        self._last_sample_t = time.time()

        # Totalisers (m³)
        self._sd8500_total = 0.0
        self._sd6500_1_total = 0.0
        self._sd6500_2_total = 0.0

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
        """Return one synthetic sample row for all sensors."""

        now = time.time()
        dt = max(1e-3, now - self._last_sample_t)
        self._last_sample_t = now

        now_utc = datetime.now(timezone.utc)
        t_rel = now - self._t0  # seconds since start

        def wave(base: float, amp: float, period_s: float, noise: float) -> float:
            val = base + amp * math.sin(2 * math.pi * t_rel / period_s)
            val += self._rng.gauss(0.0, noise)
            return val

        # Status generator (rare WARN/ERROR)
        def status(p_warn: float = 0.01, p_error: float = 0.002) -> str:
            r = self._rng.random()
            if r < p_error:
                return "ERROR"
            if r < p_error + p_warn:
                return "WARN"
            return "OK"

        row: Dict[str, Any] = {
            "timestamp_utc": now_utc.isoformat(),
            "timestamp_unix_s": now,
        }

        # ----------------------------
        # SD8500: flow / pressure / status / temp / totaliser
        # ----------------------------
        if sd8500_port is not None:
            sd8500_flow = max(0.0, wave(base=4.0, amp=1.5, period_s=60.0, noise=0.05))
            sd8500_pressure = max(0.0, wave(base=1.2, amp=0.35, period_s=90.0, noise=0.02))
            sd8500_temp = wave(base=26.0, amp=1.2, period_s=300.0, noise=0.08)

            self._sd8500_total += sd8500_flow * dt / 3600.0  # m³

            row.update(
                {
                    "sd8500_flow_m3_h": sd8500_flow,
                    "sd8500_pressure_bar": sd8500_pressure,
                    "sd8500_status": status(),
                    "sd8500_temperature_c": sd8500_temp,
                    "sd8500_totaliser_m3": self._sd8500_total,
                }
            )

        # ----------------------------
        # SD6500 #1
        # ----------------------------
        if sd6500_1_port is not None:
            flow = max(0.0, wave(base=2.5, amp=1.0, period_s=50.0, noise=0.05))
            pressure = max(0.0, wave(base=0.9, amp=0.25, period_s=80.0, noise=0.02))
            temp = wave(base=22.7, amp=0.9, period_s=280.0, noise=0.08)

            self._sd6500_1_total += flow * dt / 3600.0

            row.update(
                {
                    "sd6500_1_flow_m3_h": flow,
                    "sd6500_1_pressure_bar": pressure,
                    "sd6500_1_status": status(p_warn=0.015, p_error=0.003),
                    "sd6500_1_temperature_c": temp,
                    "sd6500_1_totaliser_m3": self._sd6500_1_total,
                }
            )

        # ----------------------------
        # SD6500 #2
        # ----------------------------
        if sd6500_2_port is not None:
            flow = max(0.0, wave(base=3.2, amp=1.2, period_s=70.0, noise=0.05))
            pressure = max(0.0, wave(base=1.0, amp=0.30, period_s=95.0, noise=0.02))
            temp = wave(base=23.1, amp=0.8, period_s=310.0, noise=0.08)

            self._sd6500_2_total += flow * dt / 3600.0

            row.update(
                {
                    "sd6500_2_flow_m3_h": flow,
                    "sd6500_2_pressure_bar": pressure,
                    "sd6500_2_status": status(p_warn=0.012, p_error=0.002),
                    "sd6500_2_temperature_c": temp,
                    "sd6500_2_totaliser_m3": self._sd6500_2_total,
                }
            )

        # ----------------------------
        # SenxTx O2 via analogue current + computed oxygen percent
        # ----------------------------
        if senxtx_port is not None:
            # 4–20 mA
            current = wave(base=12.0, amp=3.0, period_s=600.0, noise=0.05)
            current = _clamp(current, 4.0, 20.0)

            # Map 4–20 mA -> 0–25% O2 (typical span; tweak if your sensor differs)
            o2_percent = (current - 4.0) / 16.0 * 25.0
            o2_percent = _clamp(o2_percent, 0.0, 25.0)

            # Occasionally produce a NaN to test UI robustness (very rare)
            if self._rng.random() < 0.001:
                o2_percent = float("nan")

            row.update(
                {
                    "senxtx_o2_current_mA": current,
                    "senxtx_o2_oxygen_percent": o2_percent,
                }
            )

        # ----------------------------
        # Michell dewpoint via analogue current + computed dewpoint (°C)
        # ----------------------------
        if michell_port is not None:
            # 4–20 mA
            current = wave(base=10.5, amp=4.0, period_s=400.0, noise=0.05)
            current = _clamp(current, 4.0, 20.0)

            # Map 4–20 mA -> -20 .. +20 °C dewpoint (example range)
            dp_c = (current - 4.0) / 16.0 * 40.0 - 20.0

            row.update(
                {
                    "michell_dewpoint_current_mA": current,
                    "michell_dewpoint_c": dp_c,
                }
            )

        # ----------------------------
        # Dewpoint Banner #1 and #2
        # ----------------------------
        if banner_dp1_port is not None:
            humidity = _clamp(wave(base=48.0, amp=6.0, period_s=700.0, noise=0.6), 0.0, 100.0)
            temperature = wave(base=23.5, amp=1.0, period_s=500.0, noise=0.15)
            dewpoint = wave(base=12.0, amp=2.0, period_s=500.0, noise=0.2)

            row.update(
                {
                    "dewpoint_banner_1_Humidity": humidity,
                    "dewpoint_banner_1_degreeC": temperature,
                    "dewpoint_banner_1_dewpoint": dewpoint,
                }
            )

        if banner_dp2_port is not None:
            humidity = _clamp(wave(base=49.0, amp=7.0, period_s=650.0, noise=0.7), 0.0, 100.0)
            temperature = wave(base=24.0, amp=1.1, period_s=520.0, noise=0.15)
            dewpoint = wave(base=12.5, amp=2.2, period_s=520.0, noise=0.2)

            row.update(
                {
                    "dewpoint_banner_2_Humidity": humidity,
                    "dewpoint_banner_2_degreeC": temperature,
                    "dewpoint_banner_2_dewpoint": dewpoint,
                }
            )

        # ----------------------------
        # Dewpoint Banner #3
        # ----------------------------
        if banner_dp3_port is not None:
            humidity = _clamp(wave(base=50.0, amp=6.5, period_s=680.0, noise=0.65), 0.0, 100.0)
            temperature = wave(base=23.8, amp=0.95, period_s=510.0, noise=0.15)
            dewpoint = wave(base=13.0, amp=2.1, period_s=510.0, noise=0.2)

            row.update(
                {
                    "dewpoint_banner_3_Humidity": humidity,
                    "dewpoint_banner_3_degreeC": temperature,
                    "dewpoint_banner_3_dewpoint": dewpoint,
                }
            )

        # ----------------------------
        # PT100 #1..#4
        # ----------------------------
        if pt100_module_port is not None:
            base_temp = wave(base=22.2, amp=0.6, period_s=900.0, noise=0.05)
            row.update(
                {
                    "pt100_1_degC": base_temp + self._rng.gauss(0.00, 0.08),
                    "pt100_2_degC": base_temp + self._rng.gauss(0.05, 0.08),
                    "pt100_3_degC": base_temp + self._rng.gauss(-0.03, 0.08),
                    "pt100_4_degC": base_temp + self._rng.gauss(0.08, 0.08),
                }
            )

        return row
