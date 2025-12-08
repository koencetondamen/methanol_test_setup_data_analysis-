from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methanol_dashboard import config


BASE_DIR = Path(__file__).resolve().parent
EXP_DIR = BASE_DIR / "data" / "experiments"
REPORT_DIR = BASE_DIR / "data" / "reports"


def load_experiment(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load one experiment CSV + its meta json."""
    stem = csv_path.stem  # e.g. 20251208_145314_first_test
    meta_path = csv_path.with_name(stem + "_meta.json")

    df = pd.read_csv(csv_path)

    # Parse timestamps
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        df = df.sort_values("timestamp_utc")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return df, meta


def summarise_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary table (min, max, mean, std) for each numeric column."""
    numeric = df.select_dtypes(include=["number"])
    summary = pd.DataFrame(
        {
            "min": numeric.min(),
            "max": numeric.max(),
            "mean": numeric.mean(),
            "std": numeric.std(),
        }
    )
    return summary


def plot_flows(df: pd.DataFrame, outdir: Path) -> None:
    flow_fields = [
        f["field"]
        for f in config.SENSOR_FIELDS
        if "flow" in f["field"]  # crude, but matches *_flow_m3_h
    ]
    flow_fields = [f for f in flow_fields if f in df.columns]
    if not flow_fields:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for f in flow_fields:
        ax.plot(df["timestamp_utc"], df[f], label=f)

    ax.set_title("Flows vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow [m³/h]")
    ax.legend()
    ax.grid(True)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "flows_time_series.png", dpi=150)
    plt.close(fig)


def plot_dewpoints(df: pd.DataFrame, outdir: Path) -> None:
    fields = [c for c in df.columns if "dewpoint_banner" in c and c.endswith("_degC")]
    if not fields:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for f in fields:
        ax.plot(df["timestamp_utc"], df[f], label=f)

    ax.set_title("Dewpoint sensors vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Dewpoint [°C]")
    ax.legend()
    ax.grid(True)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "dewpoints_time_series.png", dpi=150)
    plt.close(fig)


def plot_pt100(df: pd.DataFrame, outdir: Path) -> None:
    fields = [c for c in df.columns if c.startswith("pt100_") and c.endswith("_degC")]
    if not fields:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for f in fields:
        ax.plot(df["timestamp_utc"], df[f], label=f)

    ax.set_title("PT100 temperatures vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]")
    ax.legend()
    ax.grid(True)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "pt100_time_series.png", dpi=150)
    plt.close(fig)


def plot_currents(df: pd.DataFrame, outdir: Path) -> None:
    fields = [c for c in df.columns if c.endswith("_current_mA")]
    if not fields:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for f in fields:
        ax.plot(df["timestamp_utc"], df[f], label=f)

    ax.set_title("Analogue currents vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Current [mA]")
    ax.legend()
    ax.grid(True)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "currents_time_series.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, outdir: Path) -> None:
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return

    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_title("Correlation matrix (numeric signals)")

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def make_report_for_csv(csv_path: Path) -> None:
    df, meta = load_experiment(csv_path)
    stem = csv_path.stem  # e.g. 20251208_145314_first_test

    print(f"\n=== Report for {stem} ===")
    if meta:
        print("Meta:", meta)

    # Basic info
    if "timestamp_utc" in df.columns and not df["timestamp_utc"].empty:
        duration = df["timestamp_utc"].max() - df["timestamp_utc"].min()
        print(f"Samples: {len(df)}, duration: {duration}")
    else:
        print(f"Samples: {len(df)} (no timestamp_utc column?)")

    summary = summarise_experiment(df)
    print("\nSummary statistics (first few rows):")
    print(summary.head(15))

    outdir = REPORT_DIR / stem

    plot_flows(df, outdir)
    plot_dewpoints(df, outdir)
    plot_pt100(df, outdir)
    plot_currents(df, outdir)
    plot_correlation(df, outdir)

    print(f"Plots written to: {outdir}")


def main() -> None:
    if not EXP_DIR.exists():
        print(f"No experiment directory found at {EXP_DIR}")
        return

    csv_files = sorted(EXP_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {EXP_DIR}")
        return

    for csv_path in csv_files:
        make_report_for_csv(csv_path)


if __name__ == "__main__":
    main()
