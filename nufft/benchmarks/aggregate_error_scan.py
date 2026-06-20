"""Aggregate per-N benchmark metrics into a single convergence plot.

Each Slurm job in the scan runs one Fourier grid size N and writes its own
``N{N}/metrics.csv``. This script collects the successful rows from every
``N*`` subfolder of a configuration's results directory and produces a combined
``error_vs_N.png`` (and ``error_vs_N.csv``) at the base of that directory.

Example:

    python benchmarks/aggregate_error_scan.py \
      --base-dir results/desc_bfield/precise_QA
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PLOT_METRICS = (
    ("rel_rms", "s-", "vector RMS"),
    ("mag_rel_rms", "^-", "|B| RMS"),
    ("rel_max", "D-", "vector max"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate N*/metrics.csv from a benchmark results directory into a "
            "combined error-vs-N convergence plot."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/desc_bfield/precise_QA"),
        help="Results directory holding N*/metrics.csv subfolders.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Configuration label for the plot title. Defaults to the base-dir name.",
    )
    return parser.parse_args()


def collect_rows(base_dir: Path) -> list[dict[str, float]]:
    """One row per N (latest wins), sorted ascending, successful runs only."""
    by_n: dict[int, dict[str, float]] = {}
    for sub in sorted(base_dir.glob("N*")):
        metrics = sub / "metrics.csv"
        if not metrics.exists():
            continue
        with metrics.open(newline="") as f:
            for raw in csv.DictReader(f):
                if raw.get("status") != "ok":
                    continue
                try:
                    record = {
                        "N": int(raw["N"]),
                        "rel_rms": float(raw["rel_rms"]),
                        "mag_rel_rms": float(raw["mag_rel_rms"]),
                        "rel_max": float(raw["rel_max"]),
                    }
                except (KeyError, ValueError):
                    continue
                by_n[record["N"]] = record
    return [by_n[n] for n in sorted(by_n)]


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = ["N", "rel_rms", "mag_rel_rms", "rel_max"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float]], path: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.array([row["N"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, style, legend in PLOT_METRICS:
        ax.loglog(N, [row[key] for row in rows], style, label=legend)
    ax.set_xlabel("Nx = Ny = Nz")
    ax.set_ylabel("relative error vs DESC B")
    ax.set_title(f"{label} DESC B comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    if not base_dir.exists():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    rows = collect_rows(base_dir)
    if not rows:
        raise SystemExit(
            f"No successful metrics rows found under {base_dir}/N*/metrics.csv."
        )

    label = args.label or base_dir.name
    csv_path = base_dir / "error_vs_N.csv"
    png_path = base_dir / "error_vs_N.png"
    write_csv(rows, csv_path)
    plot_rows(rows, png_path, label)

    print(f"Aggregated {len(rows)} grid sizes: {[row['N'] for row in rows]}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
