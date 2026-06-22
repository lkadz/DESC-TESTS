from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

from bench_config import available_configs, load_config

JOB_SCRIPT = Path("job.slurm_desc_bfield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit DESC/NUFFT stellarator B-field benchmark Slurm jobs."
    )
    parser.add_argument(
        "--config",
        default="precise_QA",
        help=(
            "Named benchmark configuration (a file in benchmarks/configurations/) "
            "or a path to a .json config. Supplies the equilibrium, coils, source "
            "model, grid, and default N values. "
            f"Available: {', '.join(available_configs()) or '(none)'}."
        ),
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="Fourier grid sizes to submit as separate jobs. Defaults to the config.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Base output directory. Each N writes to OUTDIR/N{N}. "
            "Defaults to results/desc_bfield/<config name>."
        ),
    )
    parser.add_argument(
        "--eq-file",
        type=Path,
        default=None,
        help="Optional DESC equilibrium .h5 file. Overrides the config.",
    )
    parser.add_argument(
        "--coil-file",
        type=Path,
        default=None,
        help="Optional DESC coil/external field .h5 file. Overrides the config.",
    )
    parser.add_argument(
        "--allow-missing-coils",
        action="store_true",
        help="Submit a plasma-source-only diagnostic if no coil field is available.",
    )
    parser.add_argument("--n-rho", type=int, default=16)
    parser.add_argument("--n-theta", type=int, default=32)
    parser.add_argument("--n-zeta", type=int, default=64)
    parser.add_argument(
        "--target-rho-max",
        type=float,
        default=1.0,
        help="Outermost target flux surface (1.0=LCFS; 0.95 pulls off the boundary).",
    )
    parser.add_argument(
        "--edge-taper-rho0",
        type=float,
        default=None,
        help="Taper volume current to zero from this rho to the LCFS (e.g. 0.95).",
    )
    parser.add_argument(
        "--edge-taper-shape",
        choices=("smoothstep", "smootherstep", "cosine", "quadratic"),
        default="smoothstep",
        help="Edge taper window shape.",
    )
    parser.add_argument(
        "--spectral-filter",
        choices=("none", "exponential", "lanczos", "cesaro", "raised_cosine"),
        default="none",
        help="Filter B_hat to suppress Gibbs ringing without modifying the source.",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=8,
        help="Order p of the exponential filter exp(-alpha (|k|/k_max)**p).",
    )
    parser.add_argument(
        "--lcfs-zoom",
        action="store_true",
        help=(
            "Add a near-LCFS zoom plot (BR, BZ, Bphi vs poloidal angle on a band "
            "of edge flux surfaces, DESC vs NUFFT) to look for Gibbs ringing."
        ),
    )
    parser.add_argument(
        "--lcfs-zoom-rho-min",
        type=float,
        default=0.9,
        help="Innermost flux surface of the LCFS zoom band (default 0.9).",
    )
    parser.add_argument(
        "--lcfs-zoom-n-rho",
        type=int,
        default=12,
        help="Number of flux surfaces from --lcfs-zoom-rho-min to the LCFS.",
    )
    parser.add_argument(
        "--lcfs-zoom-n-theta",
        type=int,
        default=256,
        help="Poloidal sampling of the LCFS zoom (high to resolve oscillations).",
    )
    parser.add_argument(
        "--source-scan",
        action="store_true",
        help=(
            "Submit a single job that scans source resolution at fixed box N "
            "(diagnostic 2) instead of one job per box N."
        ),
    )
    parser.add_argument(
        "--source-grids",
        nargs="+",
        default=["8,16,32", "16,32,64", "32,64,128"],
        help="Source 'n_rho,n_theta,n_zeta' triples for --source-scan.",
    )
    parser.add_argument(
        "--source-scan-n",
        type=int,
        default=64,
        help="Fixed box grid Nx=Ny=Nz for --source-scan.",
    )
    parser.add_argument(
        "--joint-scan",
        action="store_true",
        help=(
            "Submit a single job that refines box N and source resolution "
            "together (matched convergence path)."
        ),
    )
    parser.add_argument(
        "--joint-grids",
        nargs="+",
        default=[
            "16,4,8,16",
            "32,8,16,32",
            "64,16,32,64",
            "128,32,64,128",
            "256,64,128,256",
        ],
        help="Matched 'N,n_rho,n_theta,n_zeta' tuples for --joint-scan.",
    )
    parser.add_argument(
        "--time",
        default="06:00:00",
        help="Slurm wall-clock limit (HH:MM:SS). Default: 06:00:00.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Slurm partition to submit to (e.g. 'short'). Omit to use cluster default.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=("auto", "cpu", "gpu", "cuda", "rocm", "tpu"),
        default="cuda",
        help=(
            "JAX platform for the benchmark. Use 'cpu' when jax-finufft does not "
            "provide CUDA lowering in the active environment."
        ),
    )
    return parser.parse_args()


def get_queued_job_names(user: str | None = None) -> set[str]:
    user = user or getpass.getuser()
    result = subprocess.run(
        ["squeue", "-u", user, "-t", "PD,R", "-h", "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    job_names = set(filter(None, result.stdout.split("\n")))
    print(f"Found {len(job_names)} jobs already in queue (PD/R).")
    return job_names


def _path_arg(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _taper_lines(args: argparse.Namespace) -> str:
    if args.edge_taper_rho0 is None:
        return ""
    return (
        f"  --edge-taper-rho0 {args.edge_taper_rho0} \\\n"
        f"  --edge-taper-shape {args.edge_taper_shape} \\\n"
    )


def _filter_lines(args: argparse.Namespace) -> str:
    if args.spectral_filter == "none":
        return ""
    return (
        f"  --spectral-filter {args.spectral_filter} \\\n"
        f"  --filter-order {args.filter_order} \\\n"
    )


def _lcfs_zoom_lines(args: argparse.Namespace) -> str:
    if not args.lcfs_zoom:
        return ""
    return (
        f"  --lcfs-zoom \\\n"
        f"  --lcfs-zoom-rho-min {args.lcfs_zoom_rho_min} \\\n"
        f"  --lcfs-zoom-n-rho {args.lcfs_zoom_n_rho} \\\n"
        f"  --lcfs-zoom-n-theta {args.lcfs_zoom_n_theta} \\\n"
    )


def result_exists(n: int, base_outdir: Path, args: argparse.Namespace) -> bool:
    outdir = base_outdir / f"N{n}"
    metrics = outdir / "metrics.csv"
    sample = outdir / f"desc_bfield_sample_N{n}.npz"
    metadata = outdir / "run_metadata.json"
    if not (metrics.exists() and sample.exists() and metadata.exists()):
        return False
    try:
        metadata_args = json.loads(metadata.read_text()).get("args", {})
    except json.JSONDecodeError:
        return False
    return (
        metadata_args.get("config") == args.config
        and metadata_args.get("eq_file") == _path_arg(args.eq_file)
        and metadata_args.get("coil_file") == _path_arg(args.coil_file)
        and metadata_args.get("allow_missing_coils") == args.allow_missing_coils
    )


def _slurm_header(job_name: str, logfile: Path, args: argparse.Namespace) -> str:
    partition_line = f"#SBATCH --partition={args.partition}\n" if args.partition else ""
    gpu_lines = ""
    if args.jax_platform in {"gpu", "cuda"}:
        gpu_lines = "#SBATCH --gres=gpu:1\n#SBATCH --constraint=nomig\n"
    return (
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --mem=120G\n"
        f"#SBATCH --time={args.time}\n"
        f"{gpu_lines}"
        f"{partition_line}"
        f"#SBATCH -o {logfile}\n"
    )


def make_slurm_script(
    n: int, job_name: str, base_outdir: Path, args: argparse.Namespace
) -> str:
    outdir = base_outdir / f"N{n}"
    eq_file_line = f"  --eq-file {args.eq_file} \\\n" if args.eq_file else ""
    coil_file_line = f"  --coil-file {args.coil_file} \\\n" if args.coil_file else ""
    allow_missing_line = "  --allow-missing-coils \\\n" if args.allow_missing_coils else ""
    header = _slurm_header(job_name, outdir / "slurm_%j.out", args)
    return f"""#!/bin/bash
{header}

set -eo pipefail
export PS1="${{PS1-}}"

module purge
module load anaconda3/2024.10
conda activate desc-env
set -u

mkdir -p {outdir}

unset JAX_PLATFORMS
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/benchmark_desc_bfield.py \\
  --config {args.config} \\
  --desc-root auto \\
  --outdir {outdir} \\
{eq_file_line}{coil_file_line}{allow_missing_line}\
  --n-values {n} \\
  --plot-n {n} \\
  --n-rho {args.n_rho} \\
  --n-theta {args.n_theta} \\
  --n-zeta {args.n_zeta} \\
  --target-rho-max {args.target_rho_max} \\
{_taper_lines(args)}{_filter_lines(args)}{_lcfs_zoom_lines(args)}\
  --jax-platform {args.jax_platform}
"""


def make_source_scan_script(job_name: str, outdir: Path, args: argparse.Namespace) -> str:
    eq_file_line = f"  --eq-file {args.eq_file} \\\n" if args.eq_file else ""
    coil_file_line = f"  --coil-file {args.coil_file} \\\n" if args.coil_file else ""
    allow_missing_line = "  --allow-missing-coils \\\n" if args.allow_missing_coils else ""
    source_grids = " ".join(args.source_grids)
    header = _slurm_header(job_name, outdir / "slurm_%j.out", args)
    return f"""#!/bin/bash
{header}

set -eo pipefail
export PS1="${{PS1-}}"

module purge
module load anaconda3/2024.10
conda activate desc-env
set -u

mkdir -p {outdir}

unset JAX_PLATFORMS
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/benchmark_desc_bfield.py \\
  --config {args.config} \\
  --desc-root auto \\
  --outdir {outdir} \\
{eq_file_line}{coil_file_line}{allow_missing_line}\
  --source-scan \\
  --source-scan-n {args.source_scan_n} \\
  --source-grids {source_grids} \\
  --n-rho {args.n_rho} \\
  --n-theta {args.n_theta} \\
  --n-zeta {args.n_zeta} \\
  --target-rho-max {args.target_rho_max} \\
{_taper_lines(args)}{_filter_lines(args)}{_lcfs_zoom_lines(args)}\
  --jax-platform {args.jax_platform}
"""


def make_joint_scan_script(job_name: str, outdir: Path, args: argparse.Namespace) -> str:
    eq_file_line = f"  --eq-file {args.eq_file} \\\n" if args.eq_file else ""
    coil_file_line = f"  --coil-file {args.coil_file} \\\n" if args.coil_file else ""
    allow_missing_line = "  --allow-missing-coils \\\n" if args.allow_missing_coils else ""
    joint_grids = " ".join(args.joint_grids)
    header = _slurm_header(job_name, outdir / "slurm_%j.out", args)
    return f"""#!/bin/bash
{header}

set -eo pipefail
export PS1="${{PS1-}}"

module purge
module load anaconda3/2024.10
conda activate desc-env
set -u

mkdir -p {outdir}

unset JAX_PLATFORMS
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/benchmark_desc_bfield.py \\
  --config {args.config} \\
  --desc-root auto \\
  --outdir {outdir} \\
{eq_file_line}{coil_file_line}{allow_missing_line}\
  --joint-scan \\
  --joint-grids {joint_grids} \\
  --target-rho-max {args.target_rho_max} \\
{_taper_lines(args)}{_filter_lines(args)}{_lcfs_zoom_lines(args)}\
  --keep-going \\
  --jax-platform {args.jax_platform}
"""


def submit_single_job(job_name: str, outdir: Path, script: str, queued_jobs: set[str]) -> None:
    if job_name in queued_jobs:
        print(f"Skipping (job in queue/running): {job_name}")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    JOB_SCRIPT.write_text(script)
    result = subprocess.run(["sbatch", str(JOB_SCRIPT)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"!! ERROR submitting {job_name}:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout.strip())
    print(f"\tSubmitted {job_name} -> {outdir}")


def submit_job(
    n: int, queued_jobs: set[str], base_outdir: Path, args: argparse.Namespace
) -> None:
    job_name = f"desc_nufft_B_{args.config}_N{n}"
    if result_exists(n, base_outdir, args):
        print(f"Skipping {job_name}, since output files exist.")
        return
    if job_name in queued_jobs:
        print(f"Skipping (job in queue/running): {job_name}")
        return

    outdir = base_outdir / f"N{n}"
    outdir.mkdir(parents=True, exist_ok=True)
    JOB_SCRIPT.write_text(make_slurm_script(n, job_name, base_outdir, args))

    result = subprocess.run(
        ["sbatch", str(JOB_SCRIPT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("\n" + "=" * 30)
        print(f"!! ERROR submitting job: {job_name}")
        print("!! Slurm Error Message:")
        print(result.stderr)
        print("=" * 30)
        print("\nStopping submission script. You likely hit a job quota or allocation issue.")
        sys.exit(1)

    print(result.stdout.strip())
    print(f"\tSubmitted {job_name}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_name = config.get("name", args.config)
    n_values = args.n_values or config.get("n_values", [64, 128, 256])
    base_outdir = args.outdir or (Path("results/desc_bfield") / config_name)

    os.makedirs(base_outdir, exist_ok=True)
    queued_jobs = get_queued_job_names()

    # Distinct --outdir per run (e.g. baseline vs taper cases) must also yield
    # distinct job names, else submit_single_job skips the later ones as queue
    # duplicates and they overwrite each other's output.
    run_tag = base_outdir.name

    if args.source_scan:
        outdir = base_outdir / "source_scan"
        job_name = f"desc_nufft_B_{args.config}_{run_tag}_sourcescan"
        submit_single_job(
            job_name, outdir, make_source_scan_script(job_name, outdir, args), queued_jobs
        )
        print(f"\nSource-scan plot -> {outdir}/error_vs_source.png")
        return

    if args.joint_scan:
        outdir = base_outdir / "joint_scan"
        job_name = f"desc_nufft_B_{args.config}_{run_tag}_jointscan"
        submit_single_job(
            job_name, outdir, make_joint_scan_script(job_name, outdir, args), queued_jobs
        )
        print(f"\nJoint-scan plot -> {outdir}/error_vs_N.png")
        return

    for n in n_values:
        submit_job(n, queued_jobs, base_outdir, args)

    print(
        "\nWhen the jobs finish, aggregate the per-N errors with:\n"
        f"  python benchmarks/aggregate_error_scan.py --base-dir {base_outdir}"
    )


if __name__ == "__main__":
    main()
