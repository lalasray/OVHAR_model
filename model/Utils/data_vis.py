"""
Utility script to visualize the raw OVHAR sensor CSV files.

Example:
    python model/Utils/data_vis.py dataset/Participant1/Participant1/Participant1_Sensor1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sensor axes from an OVHAR CSV file."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV (e.g. dataset/Participant1/Participant1/Participant1_Sensor1.csv).",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Specific columns to plot (default: every numeric column except the time column).",
    )
    parser.add_argument(
        "--time-column",
        default="Time",
        help="Column to use for the x-axis (default: Time).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the plot to the first N rows to reduce clutter.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="If provided, save the figure to this path instead of showing it.",
    )
    return parser.parse_args()


def resolve_columns(df: pd.DataFrame, requested: Iterable[str] | None, time_col: str) -> list[str]:
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in CSV (available: {list(df.columns)})")

    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise ValueError(f"Requested columns not found: {missing}")
        return list(requested)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [col for col in numeric_cols if col != time_col]


def plot_sensor_csv(csv_path: Path, columns: Iterable[str], time_col: str, limit: int | None, save_path: Path | None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    y_columns = resolve_columns(df, columns, time_col)
    if not y_columns:
        raise ValueError("No columns left to plot after filtering.")

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in y_columns:
        ax.plot(df[time_col], df[col], label=col)

    ax.set_xlabel(time_col)
    ax.set_ylabel("Sensor reading")
    ax.set_title(f"{csv_path.stem} ({', '.join(y_columns)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    plot_sensor_csv(args.csv_path, args.columns, args.time_column, args.limit, args.save)


if __name__ == "__main__":
    main()
