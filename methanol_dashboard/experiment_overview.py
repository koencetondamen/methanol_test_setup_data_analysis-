from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config


# -----------------------------
# Sensor checklist (12 items)
# -----------------------------
SENSOR_CHECKLIST: List[Dict[str, Any]] = [
    {"name": "SD8500 flow meter", "columns_any": ["sd8500_flow_m3_h", "sd8500_pressure_bar", "sd8500_status", "sd8500_temperature_c"]},
    {"name": "SD6500 #1 flow meter", "columns_any": ["sd6500_1_flow_m3_h", "sd6500_1_pressure_bar", "sd6500_1_status", "sd6500_1_temperature_c"]},
    {"name": "SD6500 #2 flow meter", "columns_any": ["sd6500_2_flow_m3_h", "sd6500_2_pressure_bar", "sd6500_2_status", "sd6500_2_temperature_c"]},
    {"name": "SenxTx O₂", "columns_any": ["senxtx_o2_current_mA", "senxtx_o2_oxygen_percent"]},
    {"name": "Michell dewpoint (analog)", "columns_any": ["michell_dewpoint_c", "michell_dewpoint_current_mA", "dewpoint_michell_current_mA"]},
    {"name": "Banner dewpoint #1", "columns_any": ["dewpoint_banner_1_Humidity", "dewpoint_banner_1_degreeC", "dewpoint_banner_1_dewpoint"]},
    {"name": "Banner dewpoint #2", "columns_any": ["dewpoint_banner_2_Humidity", "dewpoint_banner_2_degreeC", "dewpoint_banner_2_dewpoint", "dewpoint_banner_2_degC"]},
    {"name": "PT100 #1", "columns_any": ["pt100_1_degC"]},
    {"name": "PT100 #2", "columns_any": ["pt100_2_degC"]},
    {"name": "PT100 #3", "columns_any": ["pt100_3_degC"]},
    {"name": "PT100 #4", "columns_any": ["pt100_4_degC"]},
    {"name": "Spare / future sensor", "columns_any": []},
]


EVENT_LOG_SUFFIXES = [
    "_events.csv",
    "_eventlog.csv",
    "_event_log.csv",
    "_events_log.csv",
]


@dataclass
class ExperimentBundle:
    csv_path: Path
    meta_path: Optional[Path]
    events_path: Optional[Path]
    df: pd.DataFrame
    meta: Dict[str, Any]
    events: Optional[pd.DataFrame]


# -----------------------------
# Column metadata
# -----------------------------
def _build_field_meta_from_config() -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    for f in getattr(config, "SENSOR_FIELDS", []):
        field = f.get("field")
        if not field:
            continue
        meta[field] = {
            "label": f.get("label", field),
            "unit": f.get("unit", ""),
            "entity": _infer_entity(field, f.get("label", ""), f.get("unit", "")),
        }
    return meta


def _infer_entity(col: str, label: str = "", unit: str = "") -> str:
    c = col.lower()
    l = (label or "").lower()
    u = (unit or "").lower()

    if u in ["°c", "degc", "c", "celsius"]:
        return "Temperature"
    if u in ["m³/h", "m3/h", "m^3/h"]:
        return "Flow"
    if u in ["bar"]:
        return "Pressure"
    if u in ["ma"]:
        return "Current"
    if u in ["%"]:
        if "humid" in l or "humidity" in l or "humidity" in c:
            return "Humidity"
        if "oxygen" in l or "o2" in l or "oxygen" in c or "o2" in c:
            return "Oxygen"
        return "Percent"
    if u in ["m³", "m3"]:
        return "Totaliser"

    if "flow" in l or "_flow_" in c:
        return "Flow"
    if "press" in l or "_pressure_" in c:
        return "Pressure"
    if "temp" in l or "_degc" in c or "_temperature_" in c or c.endswith("_temperature_c"):
        return "Temperature"
    if "dewpoint" in l or "dewpoint" in c:
        return "Dewpoint"
    if "humid" in l or "humidity" in c:
        return "Humidity"
    if "current" in l or "_current_" in c:
        return "Current"
    if "oxygen" in l or "o2" in l or "oxygen" in c or "o2" in c:
        return "Oxygen"
    if "totaliser" in l or "_totaliser_" in c:
        return "Totaliser"
    if c.endswith("_status") or "status" in l:
        return "Status"

    return "Other"


def _infer_unit_from_col(col: str) -> str:
    c = col.lower()
    if c.endswith("_degc") or c.endswith("_degreec") or "_temperature_" in c or c.endswith("_temperature_c"):
        return "°C"
    if c.endswith("_flow_m3_h") or "_flow_" in c:
        return "m³/h"
    if c.endswith("_pressure_bar") or "_pressure_" in c:
        return "bar"
    if c.endswith("_current_ma") or "_current_" in c:
        return "mA"
    if c.endswith("_oxygen_percent") or "humidity" in c or c.endswith("_percent"):
        return "%"
    if c.endswith("_totaliser_m3"):
        return "m³"
    return ""


def _get_col_meta(col: str, cfg_meta: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    if col in cfg_meta:
        return cfg_meta[col]
    unit = _infer_unit_from_col(col)
    entity = _infer_entity(col, label=col, unit=unit)
    return {"label": col, "unit": unit, "entity": entity}


# -----------------------------
# Invalid / indicator values
# -----------------------------
HUGE_SENTINEL_ABS = 1e30
SPECIFIC_SENTINELS = {3.299999965e38}


def _coerce_invalid_numeric_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].replace([float("inf"), float("-inf")], pd.NA)
    for c in num_cols:
        s = out[c]
        out.loc[s.abs() > HUGE_SENTINEL_ABS, c] = pd.NA
        out.loc[s.isin(list(SPECIFIC_SENTINELS)), c] = pd.NA
    return out


def _sensor_active(df: pd.DataFrame, columns_any: List[str]) -> bool:
    for c in columns_any:
        if c not in df.columns:
            continue
        series = df[c]
        if pd.api.types.is_numeric_dtype(series):
            s = pd.to_numeric(series, errors="coerce")
            s = s.replace([float("inf"), float("-inf")], pd.NA)
            s = s.mask(s.abs() > HUGE_SENTINEL_ABS, pd.NA)
            s = s.mask(s.isin(list(SPECIFIC_SENTINELS)), pd.NA)
            # 0 counts as invalid indicator for active state
            s = s.mask(s == 0, pd.NA)
            if s.notna().any():
                return True
        else:
            s = series.astype(str).str.strip().str.lower()
            s = s.replace({"nan": pd.NA, "none": pd.NA, "null": pd.NA, "": pd.NA})
            if s.notna().any():
                return True
    return False


# -----------------------------
# Loading
# -----------------------------
def _find_sidecar(csv_path: Path, suffix: str) -> Path:
    return csv_path.with_name(csv_path.stem + suffix)


def load_experiment_bundle(csv_path: Path) -> ExperimentBundle:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    meta_path = _find_sidecar(csv_path, "_meta.json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta_path = None

    events_path: Optional[Path] = None
    for suf in EVENT_LOG_SUFFIXES:
        p = _find_sidecar(csv_path, suf)
        if p.exists():
            events_path = p
            break

    df = pd.read_csv(csv_path)

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.sort_values("timestamp_utc")
    if "timestamp_unix_s" in df.columns:
        df["timestamp_unix_s"] = pd.to_numeric(df["timestamp_unix_s"], errors="coerce")

    df = _coerce_invalid_numeric_sentinels(df)

    events_df: Optional[pd.DataFrame] = None
    if events_path is not None:
        events_df = pd.read_csv(events_path)
        if "timestamp_unix_s" in events_df.columns:
            events_df["timestamp_unix_s"] = pd.to_numeric(events_df["timestamp_unix_s"], errors="coerce")
            events_df["timestamp_utc"] = pd.to_datetime(events_df["timestamp_unix_s"], unit="s", utc=True, errors="coerce")
        preferred = [c for c in ["timestamp_ams", "timestamp_unix_s", "timestamp_utc", "event"] if c in events_df.columns]
        rest = [c for c in events_df.columns if c not in preferred]
        events_df = events_df[preferred + rest]

    return ExperimentBundle(
        csv_path=csv_path,
        meta_path=meta_path,
        events_path=events_path,
        df=df,
        meta=meta,
        events=events_df,
    )


# -----------------------------
# Overview computation
# -----------------------------
def _format_td(td: timedelta) -> str:
    total_s = int(td.total_seconds())
    if total_s < 0:
        total_s = 0
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _get_experiment_time_bounds(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if "timestamp_utc" not in df.columns:
        return None, None
    ts = df["timestamp_utc"].dropna()
    if ts.empty:
        return None, None
    return ts.min(), ts.max()


def _count_invalid_indicators(series: pd.Series) -> Dict[str, int]:
    out: Dict[str, int] = {"nan": 0, "zero": 0, "huge": 0, "sentinel_3p3e38": 0}
    if series.empty:
        return out
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        out["nan"] = int(s.isna().sum())
        out["zero"] = int((s == 0).sum())
        out["huge"] = int((s.abs() > HUGE_SENTINEL_ABS).sum())
        out["sentinel_3p3e38"] = int(s.isin(list(SPECIFIC_SENTINELS)).sum())
    else:
        s = series.astype(str).str.strip().str.lower()
        out["nan"] = int(s.isin(["nan", "none", "null", ""]).sum())
    return out


def build_overview(bundle: ExperimentBundle) -> Dict[str, Any]:
    df = bundle.df
    meta = bundle.meta
    events = bundle.events

    start_utc, end_utc = _get_experiment_time_bounds(df)
    duration = (end_utc - start_utc) if (start_utc is not None and end_utc is not None) else None

    tz_ams = "Europe/Amsterdam"
    start_ams = start_utc.tz_convert(tz_ams) if start_utc is not None else None
    end_ams = end_utc.tz_convert(tz_ams) if end_utc is not None else None

    exp_name = meta.get("name") or meta.get("title") or bundle.csv_path.stem
    operator = meta.get("operator", "")
    notes = meta.get("notes", "")

    sensors_rows = []
    for s in SENSOR_CHECKLIST:
        active = _sensor_active(df, s.get("columns_any", []))
        detected = [c for c in s.get("columns_any", []) if c in df.columns]
        sensors_rows.append(
            {"sensor": s["name"], "active": active, "columns_any": ", ".join(detected) if detected else "(none found)"}
        )

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    top_nan = nan_counts.head(20)

    status_cols = [c for c in df.columns if c.lower().endswith("_status")]
    status_issues = []
    for c in status_cols:
        s = df[c].astype(str)
        s_nonnull = s[df[c].notna()]
        if s_nonnull.empty:
            continue
        bad = s_nonnull[s_nonnull.str.upper() != "OK"]
        if not bad.empty:
            status_issues.append({"column": c, "bad_counts": bad.value_counts().to_dict()})

    flow_cols = [c for c in df.columns if "flow" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    flow_invalids = {c: _count_invalid_indicators(df[c]) for c in flow_cols}

    events_rows = []
    if events is not None and start_utc is not None and "timestamp_utc" in events.columns:
        ev = events.dropna(subset=["timestamp_utc"]).copy()
        if not ev.empty:
            ev["timestamp_ams_calc"] = ev["timestamp_utc"].dt.tz_convert(tz_ams)
            ev["t_plus"] = (ev["timestamp_utc"] - start_utc)
            for _, r in ev.iterrows():
                t_plus = r.get("t_plus", pd.Timedelta(seconds=0))
                if pd.isna(t_plus):
                    t_plus = pd.Timedelta(seconds=0)
                events_rows.append(
                    {
                        "time_ams": r["timestamp_ams_calc"].strftime("%Y-%m-%d %H:%M:%S"),
                        "t_plus": _format_td(t_plus.to_pytimedelta()),
                        "event": str(r.get("event", "")),
                    }
                )

    return {
        "experiment_name": exp_name,
        "operator": operator,
        "notes": notes,
        "csv_path": str(bundle.csv_path),
        "meta_path": str(bundle.meta_path) if bundle.meta_path else None,
        "events_path": str(bundle.events_path) if bundle.events_path else None,
        "samples": int(len(df)),
        "start_ams": start_ams.strftime("%Y-%m-%d %H:%M:%S") if start_ams is not None else "—",
        "end_ams": end_ams.strftime("%Y-%m-%d %H:%M:%S") if end_ams is not None else "—",
        "duration": _format_td(duration.to_pytimedelta()) if duration is not None else "—",
        "sensors_rows": sensors_rows,
        "top_nan": top_nan.to_dict(),
        "status_issues": status_issues,
        "flow_invalids": flow_invalids,
        "events_rows": events_rows,
    }


# -----------------------------
# Plot utilities
# -----------------------------
def _estimate_sample_period_s(df: pd.DataFrame) -> float:
    if "timestamp_utc" not in df.columns:
        return 1.0
    ts = df["timestamp_utc"].dropna()
    if len(ts) < 3:
        return 1.0
    dt = ts.diff().dt.total_seconds().dropna()
    if dt.empty:
        return 1.0
    med = float(dt.median())
    return med if med > 0 else 1.0


def _clean_numeric_for_plot(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    s = s.mask(s.abs() > HUGE_SENTINEL_ABS, pd.NA)
    s = s.mask(s.isin(list(SPECIFIC_SENTINELS)), pd.NA)
    return s


def _apply_moving_average(y: pd.Series, window_s: float, sample_period_s: float) -> pd.Series:
    n = max(1, int(round(window_s / max(sample_period_s, 1e-9))))
    return y.rolling(window=n, min_periods=1).mean()


def _apply_ema(y: pd.Series, alpha: float) -> pd.Series:
    a = float(alpha)
    if a <= 0 or a > 1:
        raise ValueError("EMA alpha must be in (0, 1].")
    return y.ewm(alpha=a, adjust=False).mean()


def _assumed_uncertainty(unit: str, entity: str) -> float:
    u = (unit or "").strip()
    e = (entity or "").strip()
    if e == "Temperature" or u == "°C":
        return 0.2
    if e == "Pressure" or u == "bar":
        return 0.05
    if e == "Flow" or u == "m³/h":
        return 0.02
    if e == "Current" or u == "mA":
        return 0.05
    if e in ("Humidity", "Oxygen", "Percent") or u == "%":
        return 0.2
    if e == "Totaliser" or u == "m³":
        return 0.05
    return 0.0


def _pearson_corr_ignore_nan(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    a = x[m].astype(float)
    b = y[m].astype(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 0:
        return np.nan
    return float(np.sum(a * b) / denom)


def _corr_at_positive_lag_steps(ref: np.ndarray, other: np.ndarray, lag_steps: int) -> np.ndarray:
    """
    Returns correlations for lags 1..lag_steps (exactly lag_steps values).

    Definition here:
      lag k means we compare ref[t] with other[t+k] (other is delayed by k steps).
    So ref "leads" other by k samples.
    """
    n = min(len(ref), len(other))
    ref = ref[:n]
    other = other[:n]

    out = np.full((lag_steps,), np.nan, dtype=float)
    for k in range(1, lag_steps + 1):
        if n - k < 3:
            out[k - 1] = np.nan
            continue
        x = ref[: n - k]
        y = other[k:n]
        out[k - 1] = _pearson_corr_ignore_nan(x, y)
    return out


def _annotate_heatmap(ax, data: np.ndarray, cmap, vmin: float, vmax: float, fmt: str = "{:.2f}", fontsize: int = 8):
    """
    Annotate each cell with its value, choosing black/white based on background brightness.
    """
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    nrows, ncols = data.shape

    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            if not np.isfinite(val):
                text = "—"
                # use neutral color for missing
                color = "black"
            else:
                text = fmt.format(val)
                rgba = cmap(norm(val))
                # luminance heuristic
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if lum < 0.5 else "black"

            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=color)


# -----------------------------
# Tkinter UI
# -----------------------------
class Tooltip:
    def __init__(self, widget, text: str):
        import tkinter as tk

        self.widget = widget
        self.text = text
        self.tip = None
        self.tk = tk
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = self.tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = self.tk.Label(self.tip, text=self.text, justify="left", relief="solid", borderwidth=1, padx=8, pady=6)
        lbl.pack()

    def hide(self, _event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def show_overview_window(bundle: ExperimentBundle, overview: Dict[str, Any]) -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from tkinter.scrolledtext import ScrolledText

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    cfg_meta = _build_field_meta_from_config()

    df = bundle.df
    if "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc")

    events_df = bundle.events
    if events_df is not None and "timestamp_utc" in events_df.columns:
        events_df = events_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    def to_text_report() -> str:
        lines: List[str] = []
        lines.append(f"Experiment: {overview['experiment_name']}")
        if overview.get("operator"):
            lines.append(f"Operator: {overview['operator']}")
        lines.append(f"CSV: {overview['csv_path']}")
        if overview.get("meta_path"):
            lines.append(f"Meta: {overview['meta_path']}")
        if overview.get("events_path"):
            lines.append(f"Events: {overview['events_path']}")
        lines.append("")
        lines.append(f"Start (AMS): {overview['start_ams']}")
        lines.append(f"End   (AMS): {overview['end_ams']}")
        lines.append(f"Duration:    {overview['duration']}")
        lines.append(f"Samples:     {overview['samples']}")
        lines.append("")
        lines.append("Sensors (12):")
        for r in overview["sensors_rows"]:
            lines.append(f"  - [{'X' if r['active'] else ' '}] {r['sensor']}  (cols: {r['columns_any']})")
        lines.append("")
        lines.append("Flow invalid indicators (NaN / 0 / sentinels):")
        if overview["flow_invalids"]:
            for k, v in overview["flow_invalids"].items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  (no flow columns detected)")
        lines.append("")
        lines.append("Status issues (non-OK):")
        if overview["status_issues"]:
            for issue in overview["status_issues"]:
                lines.append(f"  - {issue['column']}: {issue['bad_counts']}")
        else:
            lines.append("  (none detected)")
        lines.append("")
        lines.append("Columns with NaNs (top):")
        if overview["top_nan"]:
            for k, v in list(overview["top_nan"].items())[:20]:
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  (none detected)")
        lines.append("")
        lines.append("Events:")
        if overview["events_rows"]:
            for e in overview["events_rows"]:
                lines.append(f"  - {e['time_ams']}  (t+ {e['t_plus']}): {e['event']}")
        else:
            lines.append("  (no event log found or no parsable timestamps)")
        return "\n".join(lines)

    # Build column groups by entity
    def build_entity_map() -> Dict[str, List[str]]:
        entity_map: Dict[str, List[str]] = {}
        for c in df.columns:
            if c in ("timestamp_utc", "timestamp_unix_s"):
                continue
            meta = _get_col_meta(c, cfg_meta)
            ent = meta["entity"]
            if ent == "Status":
                continue
            entity_map.setdefault(ent, []).append(c)
        for ent in entity_map:
            entity_map[ent] = sorted(entity_map[ent])
        return dict(sorted(entity_map.items(), key=lambda kv: kv[0]))

    entity_map = build_entity_map()
    entity_names = list(entity_map.keys())

    def listable_numeric_columns() -> List[str]:
        cols = []
        for c in df.columns:
            if c in ("timestamp_utc", "timestamp_unix_s"):
                continue
            if _get_col_meta(c, cfg_meta)["entity"] == "Status":
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                cols.append(c)
        return sorted(set(cols))

    # Root
    root = tk.Tk()
    root.title(f"Experiment Overview — {overview['experiment_name']}")
    root.geometry("1250x850")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=10, pady=10)

    # =========================
    # Summary tab
    # =========================
    tab_summary = ttk.Frame(nb)
    nb.add(tab_summary, text="Summary")

    frm = ttk.Frame(tab_summary)
    frm.pack(fill="x", padx=10, pady=10)

    def add_row(label: str, value: str, row: int) -> None:
        ttk.Label(frm, text=label, width=16).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Label(frm, text=value).grid(row=row, column=1, sticky="w", pady=4)

    add_row("Experiment", overview["experiment_name"], 0)
    add_row("Operator", overview.get("operator") or "—", 1)
    add_row("Start (AMS)", overview["start_ams"], 2)
    add_row("End (AMS)", overview["end_ams"], 3)
    add_row("Duration", overview["duration"], 4)
    add_row("Samples", str(overview["samples"]), 5)
    add_row("CSV", overview["csv_path"], 6)
    add_row("Meta", overview.get("meta_path") or "—", 7)
    add_row("Events", overview.get("events_path") or "—", 8)

    if overview.get("notes"):
        ttk.Label(tab_summary, text="Notes:").pack(anchor="w", padx=10, pady=(10, 0))
        notes = ScrolledText(tab_summary, height=6, wrap="word")
        notes.pack(fill="x", padx=10, pady=(0, 10))
        notes.insert("1.0", overview["notes"])
        notes.configure(state="disabled")

    # =========================
    # Sensors tab
    # =========================
    tab_sensors = ttk.Frame(nb)
    nb.add(tab_sensors, text="Sensors")

    tree_s = ttk.Treeview(tab_sensors, columns=("name", "active", "cols"), show="headings")
    tree_s.heading("name", text="Name")
    tree_s.heading("active", text="Active")
    tree_s.heading("cols", text="Detected Columns")
    tree_s.column("name", width=260, anchor="w")
    tree_s.column("active", width=80, anchor="center")
    tree_s.column("cols", width=820, anchor="w")
    tree_s.pack(fill="both", expand=True, padx=10, pady=10)

    for r in overview["sensors_rows"]:
        tree_s.insert("", "end", values=(r["sensor"], "YES" if r["active"] else "NO", r["columns_any"]))

    # =========================
    # Data quality tab
    # =========================
    tab_quality = ttk.Frame(nb)
    nb.add(tab_quality, text="Data quality")

    txt = ScrolledText(tab_quality, wrap="word")
    txt.pack(fill="both", expand=True, padx=10, pady=10)
    txt.insert("1.0", to_text_report())
    txt.configure(state="disabled")

    # =========================
    # Events tab
    # =========================
    tab_events = ttk.Frame(nb)
    nb.add(tab_events, text="Events")

    tree_e = ttk.Treeview(tab_events, columns=("time_ams", "t_plus", "event"), show="headings")
    tree_e.heading("time_ams", text="Time (AMS)")
    tree_e.heading("t_plus", text="t+ (into experiment)")
    tree_e.heading("event", text="Event")
    tree_e.column("time_ams", width=170, anchor="w")
    tree_e.column("t_plus", width=140, anchor="w")
    tree_e.column("event", width=820, anchor="w")
    tree_e.pack(fill="both", expand=True, padx=10, pady=10)

    if overview["events_rows"]:
        for e in overview["events_rows"]:
            tree_e.insert("", "end", values=(e["time_ams"], e["t_plus"], e["event"]))
    else:
        tree_e.insert("", "end", values=("—", "—", "No event log found / no parsable timestamps"))

    # =========================
    # Timeseries tab
    # =========================
    tab_ts = ttk.Frame(nb)
    nb.add(tab_ts, text="Timeseries")

    ts_controls = ttk.Frame(tab_ts)
    ts_controls.pack(fill="x", padx=10, pady=10)

    ttk.Label(ts_controls, text="Y-entity:").grid(row=0, column=0, sticky="w")
    ts_entity = tk.StringVar(value=entity_names[0] if entity_names else "")
    ts_entity_cb = ttk.Combobox(ts_controls, textvariable=ts_entity, values=entity_names, state="readonly", width=22)
    ts_entity_cb.grid(row=0, column=1, sticky="w", padx=(6, 18))

    ttk.Label(ts_controls, text="Columns (multi-select):").grid(row=0, column=2, sticky="w")
    ts_list = tk.Listbox(ts_controls, selectmode="extended", height=6, exportselection=False, width=55)
    ts_list.grid(row=0, column=3, sticky="w", padx=(6, 18))

    def refresh_ts_columns(*_):
        ts_list.delete(0, tk.END)
        ent = ts_entity.get()
        cols = entity_map.get(ent, [])
        for c in cols:
            meta = _get_col_meta(c, cfg_meta)
            unit = meta["unit"]
            ts_list.insert(tk.END, f"{c}   ({unit})")
        ts_list._cols = cols  # type: ignore[attr-defined]

    ts_entity_cb.bind("<<ComboboxSelected>>", refresh_ts_columns)
    refresh_ts_columns()

    smooth_frame = ttk.Frame(tab_ts)
    smooth_frame.pack(fill="x", padx=10, pady=(0, 8))

    ts_ma_on = tk.BooleanVar(value=False)
    ts_ma_win = tk.StringVar(value="5")
    ts_ema_on = tk.BooleanVar(value=False)
    ts_ema_alpha = tk.StringVar(value="0.2")
    ts_unc_on = tk.BooleanVar(value=False)

    ts_show_events = tk.BooleanVar(value=False)
    ts_label_events = tk.BooleanVar(value=True)

    ttk.Checkbutton(smooth_frame, text="Moving average", variable=ts_ma_on).grid(row=0, column=0, sticky="w")
    ttk.Label(smooth_frame, text="Window [s]:").grid(row=0, column=1, sticky="w", padx=(8, 0))
    ttk.Entry(smooth_frame, textvariable=ts_ma_win, width=8).grid(row=0, column=2, sticky="w", padx=(6, 18))

    ttk.Checkbutton(smooth_frame, text="Exponential moving average", variable=ts_ema_on).grid(row=0, column=3, sticky="w")
    ttk.Label(smooth_frame, text="Alpha (0..1]:").grid(row=0, column=4, sticky="w", padx=(8, 0))
    ttk.Entry(smooth_frame, textvariable=ts_ema_alpha, width=8).grid(row=0, column=5, sticky="w", padx=(6, 18))

    info_lbl = ttk.Label(smooth_frame, text="ⓘ", cursor="question_arrow")
    info_lbl.grid(row=0, column=6, sticky="w")
    Tooltip(
        info_lbl,
        "Moving average: each point becomes the mean of the previous N seconds.\n"
        "EMA: each point becomes a weighted average where recent samples count more.\n"
        "Alpha controls how fast EMA responds (higher = more responsive).",
    )

    ttk.Checkbutton(smooth_frame, text="Show uncertainty (assumed)", variable=ts_unc_on).grid(row=0, column=7, sticky="w", padx=(18, 0))
    ttk.Checkbutton(smooth_frame, text="Show events", variable=ts_show_events).grid(row=0, column=8, sticky="w", padx=(18, 0))
    ttk.Checkbutton(smooth_frame, text="Label events", variable=ts_label_events).grid(row=0, column=9, sticky="w", padx=(8, 0))

    axis_frame = ttk.Frame(tab_ts)
    axis_frame.pack(fill="x", padx=10, pady=(0, 8))

    ts_x_auto = tk.BooleanVar(value=True)
    ts_x0 = tk.StringVar(value="0")
    ts_x1 = tk.StringVar(value="60")

    ts_y_auto = tk.BooleanVar(value=True)
    ts_y0 = tk.StringVar(value="")
    ts_y1 = tk.StringVar(value="")

    ttk.Checkbutton(axis_frame, text="X auto", variable=ts_x_auto).grid(row=0, column=0, sticky="w")
    ttk.Label(axis_frame, text="Manual X (t+ s) from").grid(row=0, column=1, sticky="w", padx=(8, 0))
    ttk.Entry(axis_frame, textvariable=ts_x0, width=8).grid(row=0, column=2, sticky="w", padx=(6, 0))
    ttk.Label(axis_frame, text="to").grid(row=0, column=3, sticky="w", padx=(6, 0))
    ttk.Entry(axis_frame, textvariable=ts_x1, width=8).grid(row=0, column=4, sticky="w", padx=(6, 18))

    ttk.Checkbutton(axis_frame, text="Y auto", variable=ts_y_auto).grid(row=0, column=5, sticky="w")
    ttk.Label(axis_frame, text="Manual Y from").grid(row=0, column=6, sticky="w", padx=(8, 0))
    ttk.Entry(axis_frame, textvariable=ts_y0, width=8).grid(row=0, column=7, sticky="w", padx=(6, 0))
    ttk.Label(axis_frame, text="to").grid(row=0, column=8, sticky="w", padx=(6, 0))
    ttk.Entry(axis_frame, textvariable=ts_y1, width=8).grid(row=0, column=9, sticky="w", padx=(6, 18))

    title_frame = ttk.Frame(tab_ts)
    title_frame.pack(fill="x", padx=10, pady=(0, 8))
    ts_title = tk.StringVar(value="")
    ttk.Label(title_frame, text="Title:").pack(side="left")
    ttk.Entry(title_frame, textvariable=ts_title, width=60).pack(side="left", padx=(6, 18))

    ts_fig = Figure(figsize=(9, 5), dpi=100)
    ts_ax = ts_fig.add_subplot(111)
    ts_canvas = FigureCanvasTkAgg(ts_fig, master=tab_ts)
    ts_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _draw_events_on_axis(ax, xlim=None):
        if not ts_show_events.get():
            return
        if events_df is None or "timestamp_utc" not in events_df.columns:
            return

        ev = events_df
        if xlim is not None:
            lo, hi = xlim
            ev = ev[(ev["timestamp_utc"] >= lo) & (ev["timestamp_utc"] <= hi)]

        if ev.empty:
            return

        max_labels = 30
        do_labels = ts_label_events.get() and (len(ev) <= max_labels)

        for _, r in ev.iterrows():
            t_ev = r["timestamp_utc"]
            label = str(r.get("event", "")).strip()
            ax.axvline(t_ev, linestyle="--", linewidth=1, alpha=0.5)
            if do_labels and label:
                ax.text(
                    t_ev,
                    0.98,
                    label,
                    rotation=90,
                    fontsize=8,
                    ha="right",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                )

        if ts_label_events.get() and (len(ev) > max_labels):
            ax.text(
                0.01,
                0.02,
                f"Events shown: {len(ev)} (labels hidden when > {max_labels})",
                transform=ax.transAxes,
                fontsize=9,
                ha="left",
                va="bottom",
                alpha=0.8,
            )

    def ts_plot():
        if "timestamp_utc" not in df.columns or df["timestamp_utc"].dropna().empty:
            messagebox.showerror("No time axis", "This dataset has no valid timestamp_utc column.")
            return

        ent = ts_entity.get()
        cols: List[str] = getattr(ts_list, "_cols", [])  # type: ignore[attr-defined]
        sel = list(ts_list.curselection())
        if not sel:
            messagebox.showerror("No columns selected", "Select one or more columns to plot.")
            return
        chosen_cols = [cols[i] for i in sel]

        for c in chosen_cols:
            meta = _get_col_meta(c, cfg_meta)
            if meta["entity"] != ent:
                messagebox.showerror("Entity mismatch", f"Column '{c}' is '{meta['entity']}', but you selected '{ent}'.")
                return

        ts_ax.clear()
        sample_period_s = _estimate_sample_period_s(df)
        t = df["timestamp_utc"]

        try:
            win_s = float(ts_ma_win.get()) if ts_ma_on.get() else 0.0
        except Exception:
            messagebox.showerror("Invalid moving average window", "Window must be a number (seconds).")
            return
        try:
            alpha = float(ts_ema_alpha.get()) if ts_ema_on.get() else 0.0
        except Exception:
            messagebox.showerror("Invalid EMA alpha", "Alpha must be a number in (0, 1].")
            return

        for c in chosen_cols:
            meta = _get_col_meta(c, cfg_meta)
            y = _clean_numeric_for_plot(df[c])

            if ts_ma_on.get():
                y = _apply_moving_average(y, win_s, sample_period_s)
            if ts_ema_on.get():
                try:
                    y = _apply_ema(y, alpha)
                except ValueError as e:
                    messagebox.showerror("Invalid EMA alpha", str(e))
                    return

            ts_ax.plot(t, y, label=meta["label"])

            if ts_unc_on.get():
                sigma = _assumed_uncertainty(meta["unit"], meta["entity"])
                if sigma > 0:
                    ts_ax.fill_between(t, y - sigma, y + sigma, alpha=0.15)

        start_utc, _ = _get_experiment_time_bounds(df)
        if not ts_x_auto.get():
            try:
                x0 = float(ts_x0.get())
                x1 = float(ts_x1.get())
                if start_utc is None:
                    raise ValueError("Cannot compute t+ without experiment start time.")
                lo = start_utc + pd.to_timedelta(x0, unit="s")
                hi = start_utc + pd.to_timedelta(x1, unit="s")
                ts_ax.set_xlim(lo, hi)
            except Exception as e:
                messagebox.showerror("Invalid X limits", f"X limits must be valid t+ seconds.\n\n{e}")
                return

        if not ts_y_auto.get():
            try:
                ts_ax.set_ylim(float(ts_y0.get()), float(ts_y1.get()))
            except Exception as e:
                messagebox.showerror("Invalid Y limits", f"Y limits must be numbers.\n\n{e}")
                return

        unit = _get_col_meta(chosen_cols[0], cfg_meta)["unit"] if chosen_cols else ""
        ts_ax.set_xlabel("Time (UTC)")
        ts_ax.set_ylabel(f"{ent} [{unit}]" if unit else ent)

        title = ts_title.get().strip() or f"{ent} vs time"
        ts_ax.set_title(title)

        # events overlay using visible x range if possible
        try:
            lo_num, hi_num = ts_ax.get_xlim()
            lo_dt = pd.to_datetime(mdates.num2date(lo_num), utc=True)
            hi_dt = pd.to_datetime(mdates.num2date(hi_num), utc=True)
            _draw_events_on_axis(ts_ax, xlim=(lo_dt, hi_dt))
        except Exception:
            _draw_events_on_axis(ts_ax, xlim=None)

        ts_ax.grid(True)
        ts_ax.legend(loc="best")
        ts_fig.autofmt_xdate()
        ts_canvas.draw()

    ttk.Button(title_frame, text="Plot timeseries", command=ts_plot).pack(side="left")

    # =========================
    # Scatterplot tab (X vs Y)
    # =========================
    
    tab_sc = ttk.Frame(nb)
    nb.add(tab_sc, text="Scatterplot")

    sc_controls = ttk.Frame(tab_sc)
    sc_controls.pack(fill="x", padx=10, pady=10)

    ttk.Label(sc_controls, text="X-entity:").grid(row=0, column=0, sticky="w")
    sc_x_ent = tk.StringVar(value=entity_names[0] if entity_names else "")
    sc_x_ent_cb = ttk.Combobox(sc_controls, textvariable=sc_x_ent, values=entity_names, state="readonly", width=18)
    sc_x_ent_cb.grid(row=0, column=1, sticky="w", padx=(6, 18))

    ttk.Label(sc_controls, text="X-column:").grid(row=0, column=2, sticky="w")
    sc_x_col = tk.StringVar(value="")
    sc_x_col_cb = ttk.Combobox(sc_controls, textvariable=sc_x_col, values=[], state="readonly", width=45)
    sc_x_col_cb.grid(row=0, column=3, sticky="w", padx=(6, 18))

    ttk.Label(sc_controls, text="Y-entity:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    sc_y_ent = tk.StringVar(value=entity_names[1] if len(entity_names) > 1 else (entity_names[0] if entity_names else ""))
    sc_y_ent_cb = ttk.Combobox(sc_controls, textvariable=sc_y_ent, values=entity_names, state="readonly", width=18)
    sc_y_ent_cb.grid(row=1, column=1, sticky="w", padx=(6, 18), pady=(8, 0))

    ttk.Label(sc_controls, text="Y-column:").grid(row=1, column=2, sticky="w", pady=(8, 0))
    sc_y_col = tk.StringVar(value="")
    sc_y_col_cb = ttk.Combobox(sc_controls, textvariable=sc_y_col, values=[], state="readonly", width=45)
    sc_y_col_cb.grid(row=1, column=3, sticky="w", padx=(6, 18), pady=(8, 0))

    def refresh_sc_columns():
        x_ent = sc_x_ent.get()
        y_ent = sc_y_ent.get()
        x_cols = entity_map.get(x_ent, [])
        y_cols = entity_map.get(y_ent, [])
        sc_x_col_cb["values"] = x_cols
        sc_y_col_cb["values"] = y_cols
        if x_cols and sc_x_col.get() not in x_cols:
            sc_x_col.set(x_cols[0])
        if y_cols and sc_y_col.get() not in y_cols:
            sc_y_col.set(y_cols[0])

    sc_x_ent_cb.bind("<<ComboboxSelected>>", lambda *_: refresh_sc_columns())
    sc_y_ent_cb.bind("<<ComboboxSelected>>", lambda *_: refresh_sc_columns())
    refresh_sc_columns()

    sc_unc_on = tk.BooleanVar(value=False)

    sc_axis = ttk.Frame(tab_sc)
    sc_axis.pack(fill="x", padx=10, pady=(0, 8))

    sc_x_auto = tk.BooleanVar(value=True)
    sc_x0 = tk.StringVar(value="")
    sc_x1 = tk.StringVar(value="")
    sc_y_auto = tk.BooleanVar(value=True)
    sc_y0 = tk.StringVar(value="")
    sc_y1 = tk.StringVar(value="")

    ttk.Checkbutton(sc_axis, text="X auto", variable=sc_x_auto).grid(row=0, column=0, sticky="w")
    ttk.Label(sc_axis, text="Manual X from").grid(row=0, column=1, sticky="w", padx=(8, 0))
    ttk.Entry(sc_axis, textvariable=sc_x0, width=10).grid(row=0, column=2, sticky="w", padx=(6, 0))
    ttk.Label(sc_axis, text="to").grid(row=0, column=3, sticky="w", padx=(6, 0))
    ttk.Entry(sc_axis, textvariable=sc_x1, width=10).grid(row=0, column=4, sticky="w", padx=(6, 18))

    ttk.Checkbutton(sc_axis, text="Y auto", variable=sc_y_auto).grid(row=0, column=5, sticky="w")
    ttk.Label(sc_axis, text="Manual Y from").grid(row=0, column=6, sticky="w", padx=(8, 0))
    ttk.Entry(sc_axis, textvariable=sc_y0, width=10).grid(row=0, column=7, sticky="w", padx=(6, 0))
    ttk.Label(sc_axis, text="to").grid(row=0, column=8, sticky="w", padx=(6, 0))
    ttk.Entry(sc_axis, textvariable=sc_y1, width=10).grid(row=0, column=9, sticky="w", padx=(6, 18))

    sc_title_frame = ttk.Frame(tab_sc)
    sc_title_frame.pack(fill="x", padx=10, pady=(0, 8))
    sc_title = tk.StringVar(value="")
    ttk.Label(sc_title_frame, text="Title:").pack(side="left")
    ttk.Entry(sc_title_frame, textvariable=sc_title, width=60).pack(side="left", padx=(6, 18))
    ttk.Checkbutton(sc_title_frame, text="Show uncertainty (assumed)", variable=sc_unc_on).pack(side="left")

    sc_fig = Figure(figsize=(9, 5), dpi=100)
    sc_ax = sc_fig.add_subplot(111)
    sc_canvas = FigureCanvasTkAgg(sc_fig, master=tab_sc)
    sc_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def sc_plot():
        x_ent = sc_x_ent.get()
        y_ent = sc_y_ent.get()
        x_c = sc_x_col.get()
        y_c = sc_y_col.get()
        if not x_c or not y_c:
            messagebox.showerror("Missing selection", "Select both X and Y columns.")
            return

        mx = _get_col_meta(x_c, cfg_meta)
        my = _get_col_meta(y_c, cfg_meta)

        if mx["entity"] != x_ent:
            messagebox.showerror("X mismatch", f"X column '{x_c}' is '{mx['entity']}', not '{x_ent}'.")
            return
        if my["entity"] != y_ent:
            messagebox.showerror("Y mismatch", f"Y column '{y_c}' is '{my['entity']}', not '{y_ent}'.")
            return

        x = _clean_numeric_for_plot(df[x_c]).to_numpy(dtype=float)
        y = _clean_numeric_for_plot(df[y_c]).to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            messagebox.showerror("Not enough data", "Not enough valid paired samples to plot.")
            return

        x = x[m]
        y = y[m]

        sc_ax.clear()

        if sc_unc_on.get():
            sx = _assumed_uncertainty(mx["unit"], mx["entity"])
            sy = _assumed_uncertainty(my["unit"], my["entity"])
            if sx > 0 or sy > 0:
                sc_ax.errorbar(
                    x, y,
                    xerr=sx if sx > 0 else None,
                    yerr=sy if sy > 0 else None,
                    fmt="o", markersize=3, alpha=0.8
                )
            else:
                sc_ax.scatter(x, y, s=12, alpha=0.8)
        else:
            sc_ax.scatter(x, y, s=12, alpha=0.8)

        if not sc_x_auto.get():
            try:
                sc_ax.set_xlim(float(sc_x0.get()), float(sc_x1.get()))
            except Exception as e:
                messagebox.showerror("Invalid X limits", f"X limits must be numbers.\n\n{e}")
                return

        if not sc_y_auto.get():
            try:
                sc_ax.set_ylim(float(sc_y0.get()), float(sc_y1.get()))
            except Exception as e:
                messagebox.showerror("Invalid Y limits", f"Y limits must be numbers.\n\n{e}")
                return

        x_label = f"{mx['label']} [{mx['unit']}]" if mx["unit"] else mx["label"]
        y_label = f"{my['label']} [{my['unit']}]" if my["unit"] else my["label"]
        sc_ax.set_xlabel(x_label)
        sc_ax.set_ylabel(y_label)

        title = sc_title.get().strip() or f"{mx['entity']} vs {my['entity']}"
        sc_ax.set_title(title)
        sc_ax.grid(True)

        sc_canvas.draw()

    ttk.Button(sc_title_frame, text="Plot scatter", command=sc_plot).pack(side="left")

    # =========================
    # Correlation matrix tab (annotated)
    # =========================
    tab_corr = ttk.Frame(nb)
    nb.add(tab_corr, text="Correlation matrix")

    corr_top = ttk.Frame(tab_corr)
    corr_top.pack(fill="x", padx=10, pady=10)

    ttk.Label(corr_top, text="Select signals (multi-select):").grid(row=0, column=0, sticky="w")
    corr_list = tk.Listbox(corr_top, selectmode="extended", height=8, exportselection=False, width=80)
    corr_list.grid(row=1, column=0, sticky="w", pady=(6, 0))

    corr_cols = listable_numeric_columns()
    for c in corr_cols:
        meta = _get_col_meta(c, cfg_meta)
        corr_list.insert("end", f"{c}   ({meta['entity']}{' ' + meta['unit'] if meta['unit'] else ''})")
    corr_list._cols = corr_cols  # type: ignore[attr-defined]

    corr_fig = Figure(figsize=(9, 5), dpi=100)
    corr_ax = corr_fig.add_subplot(111)
    corr_canvas = FigureCanvasTkAgg(corr_fig, master=tab_corr)
    corr_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def plot_corr_matrix():
        import matplotlib.cm as cm

        sel = list(corr_list.curselection())
        cols: List[str] = getattr(corr_list, "_cols", [])  # type: ignore[attr-defined]
        if len(sel) < 2:
            messagebox.showerror("Select at least 2", "Select at least two signals for a correlation matrix.")
            return
        chosen = [cols[i] for i in sel]

        data = pd.DataFrame({c: _clean_numeric_for_plot(df[c]) for c in chosen})
        corr = data.corr().to_numpy(dtype=float)

        corr_ax.clear()
        cmap = cm.get_cmap("coolwarm")
        im = corr_ax.imshow(corr, vmin=-1, vmax=1, aspect="auto", cmap=cmap)

        corr_ax.set_xticks(np.arange(len(chosen)))
        corr_ax.set_yticks(np.arange(len(chosen)))
        corr_ax.set_xticklabels(chosen, rotation=90, fontsize=8)
        corr_ax.set_yticklabels(chosen, fontsize=8)

        corr_ax.set_title("Correlation matrix (Pearson)")

        # annotate values
        _annotate_heatmap(corr_ax, corr, cmap=cmap, vmin=-1, vmax=1, fmt="{:.2f}", fontsize=8)

        corr_fig.colorbar(im, ax=corr_ax, fraction=0.046, pad=0.04, label="corr")

        corr_fig.tight_layout()
        corr_canvas.draw()

    ttk.Button(corr_top, text="Plot correlation matrix", command=plot_corr_matrix).grid(row=0, column=1, sticky="w", padx=(12, 0))

    # =========================
    # Cross-correlation tab (reference vs many, annotated matrix)
    # =========================
    tab_xcorr = ttk.Frame(nb)
    nb.add(tab_xcorr, text="Cross-correlation")

    xcorr_top = ttk.Frame(tab_xcorr)
    xcorr_top.pack(fill="x", padx=10, pady=10)

    all_cols = listable_numeric_columns()

    ttk.Label(xcorr_top, text="Reference signal:").grid(row=0, column=0, sticky="w")
    x_ref = tk.StringVar(value=all_cols[0] if all_cols else "")
    x_ref_cb = ttk.Combobox(xcorr_top, textvariable=x_ref, values=all_cols, state="readonly", width=45)
    x_ref_cb.grid(row=0, column=1, sticky="w", padx=(6, 18))

    ttk.Label(xcorr_top, text="Compare against (multi-select):").grid(row=1, column=0, sticky="w", pady=(10, 0))
    x_list = tk.Listbox(xcorr_top, selectmode="extended", height=8, exportselection=False, width=55)
    x_list.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

    ttk.Label(xcorr_top, text="Lag width [timesteps]:").grid(row=0, column=2, sticky="w")
    x_lag_steps = tk.StringVar(value="6")
    ttk.Entry(xcorr_top, textvariable=x_lag_steps, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

    xcorr_info = ttk.Label(xcorr_top, text="ⓘ", cursor="question_arrow")
    xcorr_info.grid(row=0, column=4, sticky="w")
    Tooltip(
        xcorr_info,
        "This computes corr(reference[t], other[t+k]) for k = 1..L.\n"
        "So larger k means the other signal is delayed by k samples vs the reference.",
    )

    xcorr_status = ttk.Label(xcorr_top, text="")
    xcorr_status.grid(row=1, column=2, columnspan=3, sticky="w", pady=(10, 0))

    def refresh_xcorr_list(*_):
        x_list.delete(0, "end")
        ref = x_ref.get()
        cols = [c for c in all_cols if c != ref]
        for c in cols:
            meta = _get_col_meta(c, cfg_meta)
            x_list.insert("end", f"{c}   ({meta['entity']}{' ' + meta['unit'] if meta['unit'] else ''})")
        x_list._cols = cols  # type: ignore[attr-defined]

    x_ref_cb.bind("<<ComboboxSelected>>", refresh_xcorr_list)
    refresh_xcorr_list()

    xcorr_fig = Figure(figsize=(9, 5), dpi=100)
    xcorr_ax = xcorr_fig.add_subplot(111)
    xcorr_canvas = FigureCanvasTkAgg(xcorr_fig, master=tab_xcorr)
    xcorr_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def plot_xcorr_matrix():
        import matplotlib.cm as cm

        ref = x_ref.get()
        if not ref:
            messagebox.showerror("Missing selection", "Select a reference signal.")
            return

        sel = list(x_list.curselection())
        cols: List[str] = getattr(x_list, "_cols", [])  # type: ignore[attr-defined]
        if len(sel) < 1:
            messagebox.showerror("No comparison signals", "Select one or more comparison signals.")
            return
        others = [cols[i] for i in sel]

        try:
            L = int(float(x_lag_steps.get()))
            if L <= 0:
                raise ValueError("Lag width must be a positive integer.")
        except Exception as e:
            messagebox.showerror("Invalid lag width", str(e))
            return

        # sample period for labeling (still “timesteps” primary)
        sample_period_s = _estimate_sample_period_s(df)

        ref_arr = _clean_numeric_for_plot(df[ref]).to_numpy(dtype=float)

        mat = np.full((len(others), L), np.nan, dtype=float)
        for r_i, other in enumerate(others):
            other_arr = _clean_numeric_for_plot(df[other]).to_numpy(dtype=float)
            mat[r_i, :] = _corr_at_positive_lag_steps(ref_arr, other_arr, lag_steps=L)

        xcorr_ax.clear()
        cmap = cm.get_cmap("coolwarm")
        im = xcorr_ax.imshow(mat, vmin=-1, vmax=1, aspect="auto", cmap=cmap)

        # x ticks: lag 1..L, optionally show seconds too
        xcorr_ax.set_xticks(np.arange(L))
        xcorr_ax.set_xticklabels([f"{k}\n({k*sample_period_s:.1f}s)" for k in range(1, L + 1)], fontsize=9)
        xcorr_ax.set_yticks(np.arange(len(others)))
        xcorr_ax.set_yticklabels(others, fontsize=9)

        xcorr_ax.set_xlabel("Lag k (timesteps)  where corr(ref[t], other[t+k])")
        xcorr_ax.set_title(f"Cross-correlation vs lag: reference = {ref}")

        _annotate_heatmap(xcorr_ax, mat, cmap=cmap, vmin=-1, vmax=1, fmt="{:.2f}", fontsize=8)

        xcorr_fig.colorbar(im, ax=xcorr_ax, fraction=0.046, pad=0.04, label="corr")

        # quick peak info across all cells
        if np.isfinite(mat).any():
            idx = np.nanargmax(np.abs(mat))
            r_i, c_i = np.unravel_index(idx, mat.shape)
            best_corr = float(mat[r_i, c_i])
            best_other = others[r_i]
            best_k = c_i + 1
            xcorr_status.configure(
                text=f"Peak |corr| = {abs(best_corr):.3f} at {best_other}, lag k={best_k} "
                     f"({best_k*sample_period_s:.2f}s), corr={best_corr:+.3f}"
            )
        else:
            xcorr_status.configure(text="Not enough valid data to compute cross-correlation matrix.")

        xcorr_fig.tight_layout()
        xcorr_canvas.draw()

    ttk.Button(xcorr_top, text="Plot cross-correlation matrix", command=plot_xcorr_matrix).grid(row=0, column=5, sticky="w")

    # =========================
    # Buttons row
    # =========================
    btns = ttk.Frame(root)
    btns.pack(fill="x", padx=10, pady=(0, 10))

    def copy_to_clipboard():
        root.clipboard_clear()
        root.clipboard_append(to_text_report())
        messagebox.showinfo("Copied", "Overview copied to clipboard.")

    ttk.Button(btns, text="Copy overview to clipboard", command=copy_to_clipboard).pack(side="left")
    ttk.Button(btns, text="Close", command=root.destroy).pack(side="right")

    root.mainloop()


# -----------------------------
# CLI
# -----------------------------
def resolve_csv_path(csv_name_or_path: str) -> Path:
    p = Path(csv_name_or_path)
    if p.exists():
        return p
    return Path(config.EXPERIMENT_DIR) / csv_name_or_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show an overview popup for a logged experiment.")
    parser.add_argument("--csv", required=False, help="CSV filename (in config.EXPERIMENT_DIR) or full path to CSV.")
    args = parser.parse_args()

    CSV_FILENAME = None  # e.g. "20251230_140546_nnn.csv"
    csv_arg = args.csv or CSV_FILENAME
    if not csv_arg:
        raise SystemExit("Provide --csv <file.csv> or set CSV_FILENAME in the script.")

    csv_path = resolve_csv_path(csv_arg)

    bundle = load_experiment_bundle(csv_path)
    overview = build_overview(bundle)
    show_overview_window(bundle, overview)


if __name__ == "__main__":
    main()
