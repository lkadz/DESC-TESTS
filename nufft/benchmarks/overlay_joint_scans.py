"""Overlay error_vs_N convergence curves from several joint-scan runs.

Each ``--run`` is ``LABEL=PATH`` where PATH points at a joint-scan ``metrics.csv``
(or the directory containing it). Produces one log-log plot of relative error
vs box points (N^3) with one curve per run, so the baseline and tapered cases
can be compared directly.

Example (after the three jobs from submit_desc_bfield_scan.py finish):

    python benchmarks/overlay_joint_scans.py \
      --metric rel_rms \
      --run baseline=results/desc_bfield/2bump_n0_0.07_n1_0.02_k_iota_-1.0/baseline \
      --run taper0.95=results/desc_bfield/2bump_n0_0.07_n1_0.02_k_iota_-1.0/taper095_smoothstep \
      --run taper0.90=results/desc_bfield/2bump_n0_0.07_n1_0.02_k_iota_-1.0/taper090_quadratic \
      --out results/desc_bfield/2bump_n0_0.07_n1_0.02_k_iota_-1.0/error_vs_N_overlay.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def find_metrics_csv(path: Path) -> Path:
    """Accept either a metrics.csv or a directory holding one (recursively)."""
    if path.is_file():
        return path
    candidates = [path / "metrics.csv", path / "joint_scan" / "metrics.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(path.rglob("metrics.csv"))
    if matches:
        return matches[0]
    raise SystemExit(f"No metrics.csv found under {path}")


def load_curve(csv_path: Path, metric: str) -> tuple[list[float], list[float]]:
    ns: list[float] = []
    ys: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            try:
                n = float(row["N"])
                y = float(row[metric])
            except (KeyError, ValueError):
                continue
            ns.append(n)
            ys.append(y)
    if not ns:
        raise SystemExit(f"No successful rows with metric {metric!r} in {csv_path}")
    order = sorted(range(len(ns)), key=lambda i: ns[i])
    return [ns[i] for i in order], [ys[i] for i in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay joint-scan error curves.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="A run to overlay; PATH is a metrics.csv or its directory. Repeatable.",
    )
    parser.add_argument(
        "--metric",
        default="rel_rms",
        choices=("rel_rms", "mag_rel_rms", "rel_max", "rel_l2"),
        help="Error column to plot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("error_vs_N_overlay.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Joint-scan convergence: edge-taper comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    print(f"metric: {args.metric}\n")
    plotted = 0
    skipped = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        # Skip runs that are missing or have no successful rows (e.g. a job that
        # is still queued or that failed) instead of aborting the whole overlay.
        try:
            csv_path = find_metrics_csv(Path(raw_path))
            ns, ys = load_curve(csv_path, args.metric)
        except SystemExit as err:
            print(f"{label:>24s}  SKIPPED ({err})")
            skipped.append(label)
            continue
        x = [n**3 for n in ns]
        ax.loglog(x, ys, "o-", label=label)
        finest = f"N={int(ns[-1])}: {args.metric}={ys[-1]:.3e}"
        print(f"{label:>24s}  {csv_path}\n{'':>24s}  {finest}")
        plotted += 1

    if plotted == 0:
        raise SystemExit("No runs had usable data; nothing to plot.")
    if skipped:
        print(f"\nSkipped {len(skipped)} run(s) with no data: {', '.join(skipped)}")

    ax.set_xlabel("N^3 = Nx * Ny * Nz (box points)")
    ax.set_ylabel(f"relative error vs DESC B ({args.metric})")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
