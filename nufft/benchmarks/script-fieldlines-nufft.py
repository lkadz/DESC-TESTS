"""Trace HBT field lines with a NUFFT Biot-Savart plasma field.

This is the NUFFT analogue of ``fieldline-tracing.ipynb``:

* DESC supplies the equilibrium volume current.
* ``nufft_biot`` turns that current into a Cartesian Biot-Savart field.
* DESC's ``poincare_plot`` traces an external-field + NUFFT-plasma wrapper.
* A fixed-step RK4 tracer remains available with ``--trace-method rk4``.

The defaults target the 2-bump flat-iota HBT cases used by the notebook:

    python script-fieldlines-nufft-biot.py 0.07 0.02 -1.0

On a Slurm cluster, the command above submits a job. The job re-runs this
script with ``--run`` to do the actual tracing on a compute node.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# nufft_biot lives at nufft/src/nufft_biot in this repo (matches how
# benchmark_desc_bfield.py puts repo_root/"src" on sys.path).
NUFFT_SRC = REPO_ROOT / "src"
np = None

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*pynvml package is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Complex dtype support in Diffrax.*",
)
# We extract LCFS/seed geometry on NFP=1 grids at explicit physical toroidal
# angles (so full-torus zeta is valid for NFP>1 equilibria). DESC warns about
# the grid/basis NFP mismatch, but since we only read R/Z/x/n_rho at points
# (no integration), the values are exact. Silence the expected warning.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Unequal number of field periods.*",
)


def configure_paths():
    for path in (SCRIPT_DIR, REPO_ROOT, NUFFT_SRC):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace HBT field lines using nufft_biot volume-current Biot-Savart "
            "instead of the virtual-casing surface integral."
        )
    )
    parser.add_argument("bump_n0", nargs="?", type=float, default=0.07)
    parser.add_argument("bump_n1", nargs="?", type=float, default=0.02)
    parser.add_argument("k_iota", nargs="?", type=float, default=-1.0)
    parser.add_argument(
        "--name",
        default=None,
        help="Explicit case name. Defaults to 2bump_n0_<n0>_n1_<n1>_k_iota_<k_iota>.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Named benchmark configuration in benchmarks/configurations/<name>/. "
            "Auto-fills --name, --eq-file (eq.h5), and --coil-file (coils.h5) so "
            "you can run e.g. '--config precise_QA' without spelling out paths. "
            "Explicit --name/--eq-file/--coil-file still override it."
        ),
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        help=(
            "Run variant tag, e.g. 'baseline', 'taper095_smoothstep', 'filter_exp16'. "
            "When --config is set, results go to results/desc_bfield/<config>/<variant>/. "
            "Ignored when --save-dir is given explicitly."
        ),
    )
    parser.add_argument(
        "--source-dir",
        default="./results-2bump-rmin-0.1-simplified-flat-iota-m4",
        help="Directory containing eq-*.h5 and coils-*.h5.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Output directory for plots, trace data, Slurm files, and logs. "
            "Defaults to local results for --local, Princeton scratch otherwise."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Worker mode: run tracing now instead of submitting a Slurm job.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run immediately in this shell instead of submitting a Slurm job.",
    )
    parser.add_argument(
        "--slurm-time",
        "--time",
        dest="slurm_time",
        default="03:00:00",
        help="Slurm wall time for submitted jobs (HH:MM:SS). Alias: --time.",
    )
    parser.add_argument(
        "--slurm-mem",
        default="16G",
        help="Slurm memory request.",
    )
    parser.add_argument(
        "--slurm-cores",
        type=int,
        default=8,
        help="Slurm CPU-core request.",
    )
    parser.add_argument(
        "--slurm-gres",
        default="gpu:1",
        help="Slurm generic resource request. Use '' to omit.",
    )
    parser.add_argument(
        "--cluster",
        default=None,
        choices=("della", "della40", "della_mig", "adroit", "stellar"),
        help=(
            "Convenience preset for Slurm partition + constraint:\n"
            "  della     -> partition '',  constraint 'nomig'         (full A100, 40 or 80GB)\n"
            "  della40   -> partition '',  constraint 'nomig&gpu40'   (full 40GB A100, shorter queue)\n"
            "  della_mig -> partition 'mig'                           (MIG slice; DESC may OOM)\n"
            "  adroit    -> partition 'gpu', constraint 'gpu80'       (full A100 80GB, non-MIG)\n"
            "  stellar   -> partition 'gpu', no constraint            (full A100, no MIG)\n"
            "Explicit --slurm-partition/--slurm-constraint override the preset."
        ),
    )
    parser.add_argument(
        "--slurm-partition",
        default=None,
        help="Slurm partition. Defaults per --cluster (else empty). Overrides the preset.",
    )
    parser.add_argument(
        "--slurm-constraint",
        default=None,
        help="Slurm node constraint. Defaults per --cluster (else empty). Overrides the preset.",
    )
    parser.add_argument(
        "--conda-env",
        default="desc-env",
        help="Conda environment activated inside the Slurm job.",
    )
    parser.add_argument(
        "--stage",
        default="fb",
        choices=("fb", "solved"),
        help="Equilibrium suffix to load: eq-{name}-{stage}.h5.",
    )
    parser.add_argument(
        "--coil-file",
        default=None,
        help="Optional explicit coils h5 file. Defaults to coils-{name}-optimized.h5.",
    )
    parser.add_argument(
        "--eq-file",
        default=None,
        help=(
            "Optional explicit equilibrium h5 file (e.g. a benchmark config's "
            "eq.h5). Overrides the eq-{name}-{stage}.h5 name convention so this "
            "tracer can run any config, not just the HBT bump cases."
        ),
    )
    parser.add_argument(
        "--spectral-filter",
        choices=("none", "exponential", "lanczos", "cesaro", "raised_cosine"),
        default="none",
        help=(
            "Filter the NUFFT plasma field B_hat to suppress Gibbs ringing from "
            "the LCFS current discontinuity. Helps exterior tracing near the "
            "LCFS (the genuine field there is smooth, the ringing is high-|k|). "
            "Adds no current bias (unlike a real-space edge taper)."
        ),
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=8,
        help="Order p of the exponential filter exp(-alpha (|k|/k_max)**p).",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=("cpu", "gpu"),
        help="DESC/JAX device selection.",
    )
    parser.add_argument(
        "--xla-mem-fraction",
        default="1.0",
        help="Value for XLA_PYTHON_CLIENT_MEM_FRACTION before importing JAX.",
    )

    parser.add_argument("--nufft-cells", type=int, default=64)
    parser.add_argument("--padding", type=float, default=2.5)
    parser.add_argument("--nufft-eps", type=float, default=1e-12)
    parser.add_argument("--source-L", type=int, default=None)
    parser.add_argument("--source-M", type=int, default=None)
    parser.add_argument("--source-N", type=int, default=None)
    parser.add_argument(
        "--field-chunk-size",
        type=int,
        default=50000,
        help="Cartesian grid points per external-field evaluation chunk.",
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
        "--plasma-only",
        action="store_true",
        help="Trace only the NUFFT plasma field; do not add optimized external coils.",
    )

    parser.add_argument("--seed-theta", "--seed-poloidal", dest="seed_theta", type=int, default=10, help="Number of poloidal seed points on the LCFS offset surface.")
    parser.add_argument("--seed-phi", type=float, default=0.0)
    parser.add_argument("--seed-offset-min", type=float, default=0.01)
    parser.add_argument("--seed-offset-max", type=float, default=0.03)
    parser.add_argument("--seed-offset-count", type=int, default=3)

    parser.add_argument("--trace-method", choices=("poincare", "rk4"), default="poincare")
    parser.add_argument("--trace-backend", choices=("auto", "jax", "scipy"), default="auto")
    parser.add_argument("--trace-order", type=int, default=3)
    parser.add_argument("--interp-method", choices=("linear", "nearest", "slinear", "cubic"), default="cubic")
    parser.add_argument(
        "--trace-external-mode",
        choices=("direct", "grid"),
        default="direct",
        help=(
            "RK4-only: how to include optimized external coils during field-line tracing. "
            "'direct' evaluates coils at each RK point; 'grid' samples them on "
            "the NUFFT box and traces the interpolated total field."
        ),
    )
    parser.add_argument("--ds", type=float, default=0.005)
    parser.add_argument("--n-steps", type=int, default=12000)
    parser.add_argument("--ntransit", type=int, default=10)
    parser.add_argument("--poincare-rtol", type=float, default=1e-6)
    parser.add_argument("--poincare-atol", type=float, default=1e-6)
    parser.add_argument("--poincare-min-step-size", type=float, default=1e-8)
    parser.add_argument("--poincare-max-steps", type=int, default=10000)

    parser.add_argument("--phi-planes", type=int, default=6)
    parser.add_argument(
        "--bounds-R",
        nargs=2,
        type=float,
        default=None,
        help=(
            "Poincare plot R window. Default: auto from the LCFS bounding box "
            "(computed from the loaded equilibrium) padded by --bounds-margin."
        ),
    )
    parser.add_argument(
        "--bounds-Z",
        nargs=2,
        type=float,
        default=None,
        help="Poincare plot Z window. Default: auto from the LCFS bounding box.",
    )
    parser.add_argument(
        "--bounds-margin",
        type=float,
        default=None,
        help=(
            "Padding (m) added on every side of the LCFS bounding box for auto "
            "plot bounds. Default: max(--seed-offset-max, 10%% of the LCFS span) "
            "so exterior seeds and the flux surfaces they trace stay in frame."
        ),
    )
    parser.add_argument("--marker-size", type=float, default=0.5)
    parser.add_argument("--skip-poincare", action="store_true")
    parser.add_argument(
        "--include-inside-plasma",
        action="store_true",
        help="Keep traced lines even if they enter the plasma. Default is to exclude them.",
    )
    parser.add_argument(
        "--inside-check-phi",
        type=int,
        default=96,
        help="Toroidal planes used to detect whether field lines enter the LCFS.",
    )
    parser.add_argument(
        "--inside-check-theta",
        type=int,
        default=128,
        help="Poloidal points per LCFS plane for the inside-plasma filter.",
    )

    parser.add_argument("--skip-quiver", action="store_true")
    parser.add_argument("--quiver-planes", type=int, default=6)
    parser.add_argument("--quiver-theta", type=int, default=30)
    parser.add_argument("--quiver-offset-min", type=float, default=0.005)
    parser.add_argument("--quiver-offset-max", type=float, default=0.07)
    parser.add_argument("--quiver-offset-count", type=int, default=8)
    parser.add_argument(
        "--quiver-R",
        nargs=2,
        type=float,
        default=None,
        help="Quiver plot R window. Default: auto from the LCFS bounding box.",
    )
    parser.add_argument(
        "--quiver-Z",
        nargs=2,
        type=float,
        default=None,
        help="Quiver plot Z window. Default: auto from the LCFS bounding box.",
    )

    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Also save Poincare data or traced field-line coordinates to an npz file.",
    )
    return parser.parse_args()


def auto_variant(args):
    """Build a variant tag from taper/filter flags if --variant was not set."""
    if args.variant != "baseline":
        return args.variant
    parts = []
    if args.edge_taper_rho0 is not None:
        rho_str = f"{args.edge_taper_rho0:g}".replace(".", "")
        parts.append(f"taper{rho_str}_{args.edge_taper_shape}")
    if args.spectral_filter != "none":
        parts.append(f"filter_{args.spectral_filter}{args.filter_order}")
    return "_".join(parts) if parts else "baseline"


def output_dir(args):
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
    elif args.config is not None:
        variant = auto_variant(args)
        save_dir = REPO_ROOT / "results" / "desc_bfield" / args.config / variant / "field_line_tracing"
    elif args.local:
        save_dir = SCRIPT_DIR / "results-fieldlines-nufft-biot-local"
    else:
        save_dir = Path("/scratch/gpfs/EKOLEMEN/hbt-ep/results-fieldlines-nufft-biot")
    if not save_dir.is_absolute():
        save_dir = Path.cwd() / save_dir
    return save_dir


def worker_cli(args):
    cli = [
        "python",
        Path(__file__).name,
        f"{args.bump_n0}",
        f"{args.bump_n1}",
        f"{args.k_iota}",
        "--run",
        "--source-dir",
        str(args.source_dir),
        "--save-dir",
        str(output_dir(args)),
        "--stage",
        args.stage,
        "--device",
        args.device,
        "--xla-mem-fraction",
        str(args.xla_mem_fraction),
        "--nufft-cells",
        str(args.nufft_cells),
        "--padding",
        str(args.padding),
        "--nufft-eps",
        str(args.nufft_eps),
        "--field-chunk-size",
        str(args.field_chunk_size),
        "--seed-theta",
        str(args.seed_theta),
        "--seed-phi",
        str(args.seed_phi),
        "--seed-offset-min",
        str(args.seed_offset_min),
        "--seed-offset-max",
        str(args.seed_offset_max),
        "--seed-offset-count",
        str(args.seed_offset_count),
        "--trace-method",
        args.trace_method,
        "--trace-backend",
        args.trace_backend,
        "--trace-order",
        str(args.trace_order),
        "--interp-method",
        args.interp_method,
        "--trace-external-mode",
        args.trace_external_mode,
        "--ds",
        str(args.ds),
        "--n-steps",
        str(args.n_steps),
        "--ntransit",
        str(args.ntransit),
        "--poincare-rtol",
        str(args.poincare_rtol),
        "--poincare-atol",
        str(args.poincare_atol),
        "--poincare-min-step-size",
        str(args.poincare_min_step_size),
        "--poincare-max-steps",
        str(args.poincare_max_steps),
        "--phi-planes",
        str(args.phi_planes),
        "--marker-size",
        str(args.marker_size),
        "--inside-check-phi",
        str(args.inside_check_phi),
        "--inside-check-theta",
        str(args.inside_check_theta),
        "--quiver-planes",
        str(args.quiver_planes),
        "--quiver-theta",
        str(args.quiver_theta),
        "--quiver-offset-min",
        str(args.quiver_offset_min),
        "--quiver-offset-max",
        str(args.quiver_offset_max),
        "--quiver-offset-count",
        str(args.quiver_offset_count),
    ]

    # Plot bounds default to None (auto from the LCFS in the worker). Only pass
    # them through when the user set them explicitly, so the worker can compute
    # config-appropriate bounds otherwise.
    if args.bounds_R is not None:
        cli.extend(["--bounds-R", str(args.bounds_R[0]), str(args.bounds_R[1])])
    if args.bounds_Z is not None:
        cli.extend(["--bounds-Z", str(args.bounds_Z[0]), str(args.bounds_Z[1])])
    if args.bounds_margin is not None:
        cli.extend(["--bounds-margin", str(args.bounds_margin)])
    if args.quiver_R is not None:
        cli.extend(["--quiver-R", str(args.quiver_R[0]), str(args.quiver_R[1])])
    if args.quiver_Z is not None:
        cli.extend(["--quiver-Z", str(args.quiver_Z[0]), str(args.quiver_Z[1])])

    if args.name:
        cli.extend(["--name", args.name])
    if args.config:
        cli.extend(["--config", args.config])
    if args.variant != "baseline":
        cli.extend(["--variant", args.variant])
    if args.coil_file:
        cli.extend(["--coil-file", str(args.coil_file)])
    if args.eq_file:
        cli.extend(["--eq-file", str(args.eq_file)])
    if args.edge_taper_rho0 is not None:
        cli.extend([
            "--edge-taper-rho0", str(args.edge_taper_rho0),
            "--edge-taper-shape", args.edge_taper_shape,
        ])
    if args.spectral_filter != "none":
        cli.extend([
            "--spectral-filter", args.spectral_filter,
            "--filter-order", str(args.filter_order),
        ])
    if args.source_L is not None:
        cli.extend(["--source-L", str(args.source_L)])
    if args.source_M is not None:
        cli.extend(["--source-M", str(args.source_M)])
    if args.source_N is not None:
        cli.extend(["--source-N", str(args.source_N)])
    if args.plasma_only:
        cli.append("--plasma-only")
    if args.skip_poincare:
        cli.append("--skip-poincare")
    if args.include_inside_plasma:
        cli.append("--include-inside-plasma")
    if args.skip_quiver:
        cli.append("--skip-quiver")
    if args.save_data:
        cli.append("--save-data")
    return cli


def apply_config(args):
    """Resolve --config into --name/--eq-file/--coil-file via bench_config.

    Explicit flags win; config only fills what the user left unset. Runs before
    the submit/run branch so the worker receives concrete paths (no --config).
    """
    if not args.config:
        return
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from bench_config import load_config

    config = load_config(args.config)
    if args.name is None:
        args.name = config["name"]
    if args.eq_file is None and config.get("eq_file") is not None:
        args.eq_file = str(config["eq_file"])
    if args.coil_file is None and config.get("coil_file") is not None:
        args.coil_file = str(config["coil_file"])
    print(
        f"Config {args.config!r}: name={args.name}, "
        f"eq_file={args.eq_file}, coil_file={args.coil_file}"
    )


CLUSTER_PRESETS = {
    # Della auto-routes GPU jobs from --gres and FORBIDS naming the 'gpu'
    # partition (sbatch errors "partition of gpu ... not allowed"), so leave
    # partition empty here. 'nomig' excludes MIG slices (1g.10gb, 3g.40gb),
    # which DESC cannot use. della40 pins the full 40GB A100 (shorter queue than
    # the 80GB/H100 nodes); gpu40 alone also matches the 3g.40gb MIG slice, so it
    # must be ANDed with nomig. Adroit/Stellar do require partition 'gpu'.
    "della": {"partition": "", "constraint": "nomig"},
    "della40": {"partition": "", "constraint": "nomig&gpu40"},
    # della_mig requests a MIG slice via Della's dedicated 'mig' partition
    # (there is no 'mig' node feature -- a constraint=mig errors with "Invalid
    # feature specification"). DESC normally can't fit on a MIG slice, so this
    # is for testing / short-queue experiments only and may OOM.
    "della_mig": {"partition": "mig", "constraint": ""},
    "adroit": {"partition": "gpu", "constraint": "gpu80"},
    "stellar": {"partition": "gpu", "constraint": ""},
}


def apply_cluster(args):
    """Fill Slurm partition/constraint from --cluster unless set explicitly.

    della  -> a full non-MIG A100 via the 'nomig' feature in the default partition.
    adroit -> the full A100 80GB node (feature 'gpu80') in the 'gpu' partition,
              avoiding the MIG '3g.20gb' slices on adroit-h11g2.
    """
    preset = CLUSTER_PRESETS.get(args.cluster, {"partition": "", "constraint": ""})
    if args.slurm_partition is None:
        args.slurm_partition = preset["partition"]
    if args.slurm_constraint is None:
        args.slurm_constraint = preset["constraint"]


def slurm_job_name(args):
    if args.name:
        safe = args.name.replace("-", "m").replace(".", "p")
        return f"fl_nufft_{safe}"
    n0 = f"{args.bump_n0:g}".replace("-", "m").replace(".", "p")
    n1 = f"{args.bump_n1:g}".replace("-", "m").replace(".", "p")
    ki = f"{args.k_iota:g}".replace("-", "m").replace(".", "p")
    return f"fl_nufft_n0_{n0}_n1_{n1}_ki_{ki}"


def submit_slurm_job(args):
    save_dir = output_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    job_name = slurm_job_name(args)
    slurm_path = SCRIPT_DIR / "job.slurm_nufft_fieldlines"
    log_path = save_dir / f"slurm_{job_name}.out"
    command = " ".join(shlex.quote(str(part)) for part in worker_cli(args))

    partition_line = f"#SBATCH --partition={args.slurm_partition}\n" if args.slurm_partition else ""
    gres_line = f"#SBATCH --gres={args.slurm_gres}\n" if args.slurm_gres else ""
    constraint_line = (
        f"#SBATCH --constraint={args.slurm_constraint}\n"
        if args.slurm_constraint
        else ""
    )
    conda_line = f"conda activate {shlex.quote(args.conda_env)}" if args.conda_env else ""
    gpu_check = ""
    if args.device == "gpu":
        gpu_check = """echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found in this job environment."
    exit 2
fi
if ! nvidia-smi -L; then
    echo "ERROR: Slurm did not provide a visible NVIDIA GPU."
    exit 2
fi
python -c 'import sys, jax; devices = jax.devices(); print("Python:", sys.executable); print("JAX:", jax.__file__); print("JAX devices before DESC:", devices); raise SystemExit(0 if any(d.platform in ("gpu", "cuda") for d in devices) else 3)'
python -c 'import warnings; warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml package is deprecated.*"); import jax, desc; print("JAX devices after DESC:", jax.devices())'
"""

    slurm_text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH -n {args.slurm_cores}
#SBATCH --mem={args.slurm_mem}
#SBATCH --time={args.slurm_time}
{partition_line}{gres_line}{constraint_line}#SBATCH -o {log_path}

module purge
module load anaconda3/2024.10
export PS1="${{PS1-}}"
{conda_line}

{gpu_check}
{command}
"""
    slurm_path.write_text(slurm_text)

    result = subprocess.run(
        ["sbatch", slurm_path.name],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to submit Slurm job.\n"
            f"Command: sbatch {slurm_path}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    print(result.stdout.strip())
    print(f"Submitted {job_name}")
    print(f"Slurm file: {slurm_path}")
    print(f"Log file:   {log_path}")
    print(f"Results:    {save_dir}")


def import_runtime(args):
    global np
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", args.xla_mem_fraction)
    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "desc_hbt_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import numpy as _np
    import matplotlib

    np = _np
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from desc import set_device

    # Match the older virtual-casing script: let DESC choose the device before
    # anything asks JAX to enumerate/initialize CUDA devices.
    set_device(args.device)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    gpu_devices = [
        device for device in jax.devices() if device.platform in ("gpu", "cuda")
    ]
    gpu_device = gpu_devices[0] if gpu_devices else None
    if args.device == "gpu":
        if not gpu_devices:
            raise RuntimeError(
                "Requested --device gpu, but JAX sees no GPU devices after DESC "
                "device selection. Check the Slurm allocation, CUDA_VISIBLE_DEVICES, "
                "and nvidia-smi near the top of this job log. For CPU debugging, "
                "rerun with --local --device cpu."
            )
        print(f"JAX GPU devices after DESC device selection: {gpu_devices}")
    from desc.backend import print_backend_info
    from desc.grid import LinearGrid
    from desc.io import load
    from desc.magnetic_fields._core import _MagneticField
    from desc.plotting import poincare_plot, plot_surfaces
    from nufft_biot.desc_interface import desc_volume_current
    from nufft_biot.embedding import embed_geometry_in_box, make_optimal_box
    from nufft_biot.field import compute_B_hat, eval_B
    from nufft_biot.utils import PeriodicFieldInterpolator, trace_field_line_rk4

    try:
        from nufft_biot.utils.fast_tracing import trace_field_line_jax

        fast_trace_error = None
    except Exception as err:  # pragma: no cover - depends on optional interpax install
        trace_field_line_jax = None
        fast_trace_error = err

    print_backend_info()

    return SimpleNamespace(
        plt=plt,
        MplPath=MplPath,
        jax=jax,
        jnp=jnp,
        LinearGrid=LinearGrid,
        load=load,
        MagneticFieldBase=_MagneticField,
        poincare_plot=poincare_plot,
        plot_surfaces=plot_surfaces,
        desc_volume_current=desc_volume_current,
        embed_geometry_in_box=embed_geometry_in_box,
        make_optimal_box=make_optimal_box,
        compute_B_hat=compute_B_hat,
        eval_B=eval_B,
        gpu_device=gpu_device,
        PeriodicFieldInterpolator=PeriodicFieldInterpolator,
        trace_field_line_rk4=trace_field_line_rk4,
        trace_field_line_jax=trace_field_line_jax,
        fast_trace_error=fast_trace_error,
    )


def resolve_existing_dir(path_arg):
    path = Path(path_arg)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    script_path = SCRIPT_DIR / path
    if cwd_path.exists():
        return cwd_path
    return script_path


def case_name(args):
    if args.name:
        return args.name
    return f"2bump_n0_{args.bump_n0}_n1_{args.bump_n1}_k_iota_{args.k_iota}"


def load_case(rt, args):
    name = case_name(args)
    source_dir = resolve_existing_dir(args.source_dir)
    save_dir = output_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    eq_path = Path(args.eq_file) if args.eq_file else source_dir / f"eq-{name}-{args.stage}.h5"
    coil_path = Path(args.coil_file) if args.coil_file else source_dir / f"coils-{name}-optimized.h5"

    if not eq_path.exists():
        available = "\n".join(f"  {p.name}" for p in sorted(source_dir.glob(f"eq-{name}-*.h5")))
        available = available or "  none found"
        raise FileNotFoundError(
            f"Could not find {eq_path}\n"
            f"Available equilibrium files for {name}:\n{available}"
        )
    if not coil_path.exists():
        raise FileNotFoundError(f"Could not find {coil_path}")

    print(f"Loading equilibrium: {eq_path}")
    eq = rt.load(eq_path)
    # eq.h5 may hold the full solve history (EquilibriaFamily); take the final
    # converged equilibrium. Matches benchmark_desc_bfield.load_equilibrium.
    if eq.__class__.__name__ == "EquilibriaFamily":
        print(f"  loaded EquilibriaFamily with {len(eq)} states; using the last.")
        eq = eq[-1]
    print(f"Loading external field: {coil_path}")
    field = rt.load(coil_path)
    return name, source_dir, save_dir, eq, field


def wrap_centered(points, box):
    lengths = np.array([box.Lx, box.Ly, box.Lz], dtype=float)
    return (np.asarray(points, dtype=float) + 0.5 * lengths) % lengths - 0.5 * lengths


def xyz_to_rpz(xyz):
    xyz = np.asarray(xyz, dtype=float)
    out = np.empty_like(xyz)
    out[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    out[:, 1] = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    out[:, 2] = xyz[:, 2]
    return out


def xyz_vec_to_rpz(vec_xyz, phi):
    vec_xyz = np.asarray(vec_xyz, dtype=float)
    phi = np.asarray(phi, dtype=float)
    c = np.cos(phi)
    s = np.sin(phi)
    out = np.empty_like(vec_xyz)
    out[:, 0] = vec_xyz[:, 0] * c + vec_xyz[:, 1] * s
    out[:, 1] = -vec_xyz[:, 0] * s + vec_xyz[:, 1] * c
    out[:, 2] = vec_xyz[:, 2]
    return out


def rpz_to_xyz(rpz):
    rpz = np.asarray(rpz, dtype=float)
    out = np.empty_like(rpz)
    c = np.cos(rpz[:, 1])
    s = np.sin(rpz[:, 1])
    out[:, 0] = rpz[:, 0] * c
    out[:, 1] = rpz[:, 0] * s
    out[:, 2] = rpz[:, 2]
    return out


def rpz_to_xyz_jnp(rt, rpz):
    c = rt.jnp.cos(rpz[:, 1])
    s = rt.jnp.sin(rpz[:, 1])
    return rt.jnp.stack((rpz[:, 0] * c, rpz[:, 0] * s, rpz[:, 2]), axis=1)


def xyz_vec_to_rpz_jnp(rt, vec_xyz, phi):
    c = rt.jnp.cos(phi)
    s = rt.jnp.sin(phi)
    return rt.jnp.stack(
        (
            vec_xyz[:, 0] * c + vec_xyz[:, 1] * s,
            -vec_xyz[:, 0] * s + vec_xyz[:, 1] * c,
            vec_xyz[:, 2],
        ),
        axis=1,
    )


def grid_chunk_positions(start, stop, box):
    idx = np.arange(start, stop, dtype=np.int64)
    plane = box.Ny * box.Nz
    ix = idx // plane
    iy = (idx // box.Nz) % box.Ny
    iz = idx % box.Nz

    pos = np.column_stack(
        (
            ix * box.Lx / box.Nx,
            iy * box.Ly / box.Ny,
            iz * box.Lz / box.Nz,
        )
    )
    return wrap_centered(pos, box)


def to_requested_jax_device(rt, *arrays):
    out = []
    for array in arrays:
        arr = rt.jnp.asarray(array)
        if rt.gpu_device is not None:
            arr = rt.jax.device_put(arr, rt.gpu_device)
        out.append(arr)
    return out


def build_plasma_field_grid(rt, eq, args):
    print("Extracting DESC volume current for NUFFT Biot-Savart...")
    X, Y, Z, Jx, Jy, Jz, w = rt.desc_volume_current(
        eq,
        L_grid=args.source_L,
        M_grid=args.source_M,
        N_grid=args.source_N,
        taper_rho0=args.edge_taper_rho0,
        taper_shape=args.edge_taper_shape,
    )
    print(f"  Source points: {len(X)}")

    print("Building NUFFT box...")
    box = rt.make_optimal_box(X, Y, Z, n_cells=args.nufft_cells, padding=args.padding)
    Xb, Yb, Zb, shift = rt.embed_geometry_in_box(X, Y, Z, box)
    print(f"  Geometry shift: {np.asarray(shift)}")

    if args.device == "gpu" and rt.gpu_device is not None:
        print(f"Placing NUFFT plasma arrays on GPU: {rt.gpu_device}")
        Xb, Yb, Zb, Jx, Jy, Jz, w = to_requested_jax_device(
            rt,
            Xb,
            Yb,
            Zb,
            Jx,
            Jy,
            Jz,
            w,
        )

    print("Computing plasma-field Fourier modes...")
    Bx_hat, By_hat, Bz_hat = rt.compute_B_hat(
        Xb,
        Yb,
        Zb,
        Jx,
        Jy,
        Jz,
        w,
        box,
        eps=args.nufft_eps,
        spectral_filter=args.spectral_filter,
        filter_order=args.filter_order,
    )

    print("Transforming plasma field to Cartesian grid...")
    Bx_grid = rt.jnp.fft.ifftn(rt.jnp.fft.ifftshift(Bx_hat) * box.N_total).real
    By_grid = rt.jnp.fft.ifftn(rt.jnp.fft.ifftshift(By_hat) * box.N_total).real
    Bz_grid = rt.jnp.fft.ifftn(rt.jnp.fft.ifftshift(Bz_hat) * box.N_total).real

    source = SimpleNamespace(
        X=np.asarray(X),
        Y=np.asarray(Y),
        Z=np.asarray(Z),
        Jx=np.asarray(Jx),
        Jy=np.asarray(Jy),
        Jz=np.asarray(Jz),
        w=np.asarray(w),
    )
    plasma_modes = (Bx_hat, By_hat, Bz_hat)
    return box, np.asarray(shift), plasma_modes, Bx_grid, By_grid, Bz_grid, source


def add_external_field_grid(rt, field, box, shift, Bx_grid, By_grid, Bz_grid, args):
    Bx_total = np.asarray(Bx_grid).copy()
    By_total = np.asarray(By_grid).copy()
    Bz_total = np.asarray(Bz_grid).copy()

    if args.plasma_only:
        print("Skipping external-field grid (--plasma-only).")
        return Bx_total, By_total, Bz_total

    print("Sampling optimized external field on the NUFFT box...")
    n_total = box.N_total
    B_ext_flat = np.empty((n_total, 3), dtype=float)
    chunk_size = max(1, int(args.field_chunk_size))

    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        local = grid_chunk_positions(start, stop, box)
        world = local + shift
        B_ext_flat[start:stop] = np.asarray(
            field.compute_magnetic_field(
                world,
                basis="xyz",
                chunk_size=chunk_size,
            )
        )
        print(f"  external field chunk {stop}/{n_total}")

    B_ext = B_ext_flat.reshape((box.Nx, box.Ny, box.Nz, 3))
    Bx_total += B_ext[..., 0]
    By_total += B_ext[..., 1]
    Bz_total += B_ext[..., 2]
    return Bx_total, By_total, Bz_total


def make_surface_offset_seeds(rt, eq, args):
    offsets = np.linspace(
        args.seed_offset_min,
        args.seed_offset_max,
        args.seed_offset_count,
    )
    grid = rt.LinearGrid(
        rho=np.array([1.0]),
        theta=args.seed_theta,
        zeta=np.array([args.seed_phi]),
        # physical seed angle; NFP=1 so any --seed-phi is valid (not just the
        # first field period) and geometry is evaluated at the true angle.
        NFP=1,
    )
    data = eq.compute(["x", "n_rho"], grid=grid, basis="xyz")
    surface = np.asarray(data["x"], dtype=float)
    normal = np.asarray(data["n_rho"], dtype=float)

    seeds = [surface + offset * normal for offset in offsets]
    seeds = np.concatenate(seeds, axis=0)
    print(f"Generated {len(seeds)} seeds from {len(offsets)} LCFS offsets.")
    return seeds


def make_external_plus_nufft_field(rt, ext_field, plasma_modes, box, shift, args):
    base_static = list(getattr(rt.MagneticFieldBase, "_static_attrs", []))

    class ExternalPlusNUFFTField(rt.MagneticFieldBase):
        _static_attrs = base_static + [
            "box",
            "plasma_only",
            "nufft_eps",
            "field_chunk_size",
        ]

        def __init__(self):
            self.ext_field = ext_field
            self.Bx_hat, self.By_hat, self.Bz_hat = plasma_modes
            self.box = box
            self.shift = rt.jnp.asarray(shift)
            self.plasma_only = args.plasma_only
            self.nufft_eps = args.nufft_eps
            self.field_chunk_size = args.field_chunk_size

        def compute_magnetic_field(
            self,
            coords,
            params=None,
            basis="rpz",
            source_grid=None,
            transforms=None,
            chunk_size=None,
        ):
            basis = basis.lower()
            coords = rt.jnp.atleast_2d(coords)

            if basis == "xyz":
                coords_xyz = coords
                phi = rt.jnp.arctan2(coords[:, 1], coords[:, 0])
            elif basis == "rpz":
                coords_xyz = rpz_to_xyz_jnp(rt, coords)
                phi = coords[:, 1]
            else:
                raise ValueError(f"Unsupported basis {basis!r}; expected 'rpz' or 'xyz'.")

            targets_local = coords_xyz - self.shift
            bx_p, by_p, bz_p = rt.eval_B(
                self.Bx_hat,
                self.By_hat,
                self.Bz_hat,
                targets_local,
                self.box,
                eps=self.nufft_eps,
            )
            B_plasma_xyz = rt.jnp.stack((bx_p, by_p, bz_p), axis=1)

            if basis == "rpz":
                B_plasma = xyz_vec_to_rpz_jnp(rt, B_plasma_xyz, phi)
            else:
                B_plasma = B_plasma_xyz
            B_plasma = rt.jnp.real(B_plasma)

            if self.plasma_only:
                return B_plasma

            B_external = self.ext_field.compute_magnetic_field(
                coords,
                basis=basis,
                chunk_size=chunk_size if chunk_size is not None else self.field_chunk_size,
            )
            return rt.jnp.real(B_external + B_plasma)

        def compute_magnetic_vector_potential(self, *args, **kwargs):
            raise NotImplementedError

    return ExternalPlusNUFFTField()


def choose_trace_backend(rt, args, direct_external=False):
    if direct_external:
        if args.trace_backend == "jax":
            print(
                "Direct external-field tracing needs Python coil evaluations; "
                "using scipy RK4 instead of JAX."
            )
        return "scipy"
    if args.trace_backend == "jax":
        if rt.trace_field_line_jax is None:
            raise ImportError(f"JAX fast tracer unavailable: {rt.fast_trace_error}")
        return "jax"
    if args.trace_backend == "scipy":
        return "scipy"
    if rt.trace_field_line_jax is None:
        print(f"JAX fast tracer unavailable ({rt.fast_trace_error}); falling back to scipy RK4.")
        return "scipy"
    return "jax"


class PlasmaGridPlusDirectExternalField:
    """Evaluate plasma from the NUFFT grid and coils directly at trace points."""

    def __init__(self, rt, field, Bx, By, Bz, box, shift, args):
        self.field = field
        self.box = box
        self.shift = np.asarray(shift, dtype=float)
        self.args = args
        self.plasma_interp = rt.PeriodicFieldInterpolator(
            Bx,
            By,
            Bz,
            box,
            method=args.interp_method,
        )

    def _plasma_field(self, points_grid):
        return np.vstack([self.plasma_interp(point.copy()) for point in points_grid])

    def __call__(self, points_grid):
        points_grid = np.asarray(points_grid, dtype=float)
        scalar_input = points_grid.ndim == 1
        points_grid = np.atleast_2d(points_grid)

        B = self._plasma_field(points_grid)
        if not self.args.plasma_only:
            points_world = wrap_centered(points_grid, self.box) + self.shift
            B += np.asarray(
                self.field.compute_magnetic_field(
                    points_world,
                    basis="xyz",
                    chunk_size=max(1, min(self.args.field_chunk_size, len(points_world))),
                )
            )

        return B[0] if scalar_input else B


def normalized_directions(B, min_norm=1e-12):
    norms = np.linalg.norm(B, axis=1)
    ok = np.isfinite(norms) & (norms > min_norm)
    directions = np.zeros_like(B)
    directions[ok] = B[ok] / norms[ok, None]
    return directions, ok


def trace_field_lines_rk4_batched(B_func, seeds_grid, box, *, ds, n_steps):
    seeds_grid = np.asarray(seeds_grid, dtype=float)
    if seeds_grid.ndim != 2 or seeds_grid.shape[1] != 3:
        raise ValueError("seeds_grid must have shape (n_lines, 3)")

    lengths = np.array([box.Lx, box.Ly, box.Lz], dtype=float)
    lines = np.zeros((len(seeds_grid), n_steps, 3), dtype=float)
    lines[:, 0, :] = np.mod(seeds_grid, lengths)
    active = np.ones(len(seeds_grid), dtype=bool)

    for step in range(n_steps - 1):
        curr = np.mod(lines[:, step, :], lengths)
        lines[:, step, :] = curr
        next_pos = curr.copy()
        idx = np.flatnonzero(active)

        if len(idx):
            curr_i = curr[idx]

            k1_dir, ok = normalized_directions(B_func(curr_i))
            idx1 = idx[ok]
            inactive = idx[~ok]
            active[inactive] = False

            if len(idx1):
                k1 = k1_dir[ok] * ds
                p2 = curr[idx1] + 0.5 * k1
                k2_dir, ok = normalized_directions(B_func(p2))
                idx2 = idx1[ok]
                active[idx1[~ok]] = False

                if len(idx2):
                    k1 = k1[ok]
                    k2 = k2_dir[ok] * ds
                    p3 = curr[idx2] + 0.5 * k2
                    k3_dir, ok = normalized_directions(B_func(p3))
                    idx3 = idx2[ok]
                    active[idx2[~ok]] = False

                    if len(idx3):
                        k1 = k1[ok]
                        k2 = k2[ok]
                        k3 = k3_dir[ok] * ds
                        p4 = curr[idx3] + k3
                        k4_dir, ok = normalized_directions(B_func(p4))
                        idx4 = idx3[ok]
                        active[idx3[~ok]] = False

                        if len(idx4):
                            k1 = k1[ok]
                            k2 = k2[ok]
                            k3 = k3[ok]
                            k4 = k4_dir[ok] * ds
                            next_pos[idx4] = (
                                curr[idx4] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                            )

        lines[:, step + 1, :] = np.mod(next_pos, lengths)
        if (step + 1) % max(1, n_steps // 10) == 0:
            print(
                f"  traced RK step {step + 1}/{n_steps - 1} "
                f"({np.count_nonzero(active)}/{len(active)} active lines)"
            )
        if not np.any(active):
            lines[:, step + 2 :, :] = lines[:, step + 1 : step + 2, :]
            print(f"  stopped early at RK step {step + 1}: all lines inactive.")
            break

    return lines


def trace_lines(rt, field, Bx, By, Bz, box, shift, seeds_world, args):
    direct_external = args.trace_external_mode == "direct" and not args.plasma_only
    backend = choose_trace_backend(rt, args, direct_external=direct_external)
    print(f"Tracing field lines with {backend} backend...")

    field_lines = []
    lengths = np.array([box.Lx, box.Ly, box.Lz], dtype=float)

    if direct_external:
        print("  plasma: NUFFT-grid interpolation")
        print("  external coils: direct DESC evaluation at RK substeps")
        trace_field = PlasmaGridPlusDirectExternalField(
            rt,
            field,
            np.asarray(Bx),
            np.asarray(By),
            np.asarray(Bz),
            box,
            shift,
            args,
        )
        seeds_local = np.asarray(seeds_world, dtype=float) - shift
        seeds_grid = np.mod(seeds_local, lengths)
        line_grid = trace_field_lines_rk4_batched(
            trace_field,
            seeds_grid,
            box,
            ds=args.ds,
            n_steps=args.n_steps,
        )
        field_lines = wrap_centered(line_grid, box) + shift
        for i, line in enumerate(field_lines):
            line_local = line - shift
            if np.any(np.abs(line_local) > 0.45 * lengths):
                print(f"  warning: line {i} approaches the periodic box boundary.")
        return field_lines

    if args.trace_external_mode == "grid" and not args.plasma_only:
        print("  external coils: interpolated from the NUFFT box grid")

    if backend == "jax":
        Bx_j = rt.jnp.asarray(Bx)
        By_j = rt.jnp.asarray(By)
        Bz_j = rt.jnp.asarray(Bz)
        for i, seed_world in enumerate(seeds_world):
            seed_local = np.asarray(seed_world, dtype=float) - shift
            line_local = np.asarray(
                rt.trace_field_line_jax(
                    Bx_j,
                    By_j,
                    Bz_j,
                    rt.jnp.asarray(seed_local),
                    box.Lx,
                    box.Ly,
                    box.Lz,
                    ds=args.ds,
                    n_steps=args.n_steps,
                    order=args.trace_order,
                )
            )
            if np.any(np.abs(line_local) > 0.45 * lengths):
                print(f"  warning: line {i} approaches the periodic box boundary.")
            field_lines.append(wrap_centered(line_local, box) + shift)
            print(f"  traced seed {i + 1}/{len(seeds_world)}")
    else:
        interp = rt.PeriodicFieldInterpolator(Bx, By, Bz, box, method=args.interp_method)
        for i, seed_world in enumerate(seeds_world):
            seed_local = np.asarray(seed_world, dtype=float) - shift
            seed_grid = np.mod(seed_local, lengths)
            line_grid = rt.trace_field_line_rk4(
                interp,
                seed_grid,
                box,
                ds=args.ds,
                n_steps=args.n_steps,
            )
            field_lines.append(wrap_centered(line_grid, box) + shift)
            print(f"  traced seed {i + 1}/{len(seeds_world)}")

    return np.asarray(field_lines)


def poincare_crossings(field_lines, nplanes):
    planes = np.linspace(0.0, 2.0 * np.pi, nplanes, endpoint=False)
    crossings = [[] for _ in range(nplanes)]

    for seed_idx, line in enumerate(field_lines):
        x = line[:, 0]
        y = line[:, 1]
        z = line[:, 2]
        radius = np.sqrt(x**2 + y**2)
        phi = np.unwrap(np.arctan2(y, x))

        for i in range(len(phi) - 1):
            p0 = phi[i]
            p1 = phi[i + 1]
            if not np.isfinite(p0 + p1) or abs(p1 - p0) < 1e-14:
                continue

            lo = min(p0, p1)
            hi = max(p0, p1)
            for plane_idx, plane in enumerate(planes):
                k_min = int(np.ceil((lo - plane) / (2.0 * np.pi)))
                k_max = int(np.floor((hi - plane) / (2.0 * np.pi)))
                for k in range(k_min, k_max + 1):
                    target = plane + 2.0 * np.pi * k
                    if target <= lo or target > hi:
                        continue
                    alpha = (target - p0) / (p1 - p0)
                    if alpha < 0.0 or alpha > 1.0:
                        continue
                    R_cross = (1.0 - alpha) * radius[i] + alpha * radius[i + 1]
                    Z_cross = (1.0 - alpha) * z[i] + alpha * z[i + 1]
                    crossings[plane_idx].append((R_cross, Z_cross, seed_idx))

    return planes, crossings


def build_lcfs_paths(rt, eq, args):
    n_phi = max(1, int(args.inside_check_phi))
    n_theta = max(8, int(args.inside_check_theta))
    phi_values = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    paths = []

    print(
        "Building LCFS polygons for inside-plasma filter "
        f"({n_phi} phi planes, {n_theta} theta points)..."
    )
    for phi in phi_values:
        grid = rt.LinearGrid(
            rho=np.array([1.0]),
            theta=n_theta,
            zeta=np.array([phi]),
            # phi is a physical toroidal angle over the full torus; the eq is
            # periodic, so NFP=1 evaluates geometry correctly and avoids DESC's
            # "LinearGrid should be defined on 1 field period" restriction (NFP>1).
            NFP=1,
        )
        data = eq.compute(["R", "Z"], grid=grid)
        polygon = np.column_stack(
            (
                np.asarray(data["R"], dtype=float),
                np.asarray(data["Z"], dtype=float),
            )
        )
        paths.append(rt.MplPath(polygon, closed=True))

    return phi_values, paths


def line_enters_lcfs(line, lcfs_paths):
    _, paths = lcfs_paths
    n_phi = len(paths)
    coords = xyz_to_rpz(line)
    phi_idx = np.mod(
        np.floor(coords[:, 1] / (2.0 * np.pi) * n_phi + 0.5).astype(int),
        n_phi,
    )
    rz = coords[:, [0, 2]]

    for idx in np.unique(phi_idx):
        mask = phi_idx == idx
        if np.any(paths[idx].contains_points(rz[mask])):
            return True
    return False


def filter_lines_outside_plasma(rt, eq, field_lines, seeds_world, args):
    if args.include_inside_plasma:
        return field_lines, seeds_world
    if len(field_lines) == 0:
        return field_lines, seeds_world

    lcfs_paths = build_lcfs_paths(rt, eq, args)
    keep = np.array(
        [not line_enters_lcfs(line, lcfs_paths) for line in field_lines],
        dtype=bool,
    )
    removed = len(keep) - np.count_nonzero(keep)
    print(
        f"Inside-plasma filter kept {np.count_nonzero(keep)}/{len(keep)} lines "
        f"and removed {removed}."
    )

    if not np.any(keep):
        print("  warning: every traced line entered the plasma; Poincare plot will be empty.")
    return field_lines[keep], seeds_world[keep]


def normalize_poincare_data(data, args):
    R = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    if R.ndim == 3:
        return R, Z
    if R.ndim == 2:
        npoints, nseeds = R.shape
        nplanes = int(args.phi_planes)
        if npoints % nplanes != 0:
            raise ValueError(
                f"Cannot reshape poincare data with shape {R.shape} into "
                f"{nplanes} phi planes."
            )
        return R.reshape((-1, nplanes, nseeds)), Z.reshape((-1, nplanes, nseeds))
    raise ValueError(f"Unexpected poincare data shape: R{R.shape}, Z{Z.shape}")


def filter_poincare_data_outside_plasma(rt, eq, data, seeds_world, args):
    if args.include_inside_plasma:
        return data, seeds_world

    R, Z = normalize_poincare_data(data, args)
    _, paths = build_lcfs_paths(rt, eq, args)
    npaths = len(paths)
    nplanes = R.shape[1]
    nseeds = R.shape[2]
    keep = np.ones(nseeds, dtype=bool)

    # build_lcfs_paths samples the full torus [0, 2*pi); the punctures live at
    # the section angles (one field period). Map each plane to the LCFS polygon
    # at the SAME physical phi, else the filter checks against the wrong slice.
    section_phi = plane_angles(eq, nplanes)

    for seed_idx in range(nseeds):
        for plane_idx in range(nplanes):
            path_idx = int(round(section_phi[plane_idx] / (2.0 * np.pi) * npaths)) % npaths
            rz = np.column_stack((R[:, plane_idx, seed_idx], Z[:, plane_idx, seed_idx]))
            finite = np.isfinite(rz).all(axis=1)
            if np.any(paths[path_idx].contains_points(rz[finite])):
                keep[seed_idx] = False
                break

    removed = len(keep) - np.count_nonzero(keep)
    print(
        f"Inside-plasma filter kept {np.count_nonzero(keep)}/{len(keep)} Poincare seeds "
        f"and removed {removed}."
    )
    if not np.any(keep):
        print("  warning: every Poincare seed entered the plasma; plot will be empty.")

    filtered = dict(data)
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[-1] == nseeds:
            filtered[key] = arr[..., keep]
        else:
            filtered[key] = value
    return filtered, seeds_world[keep]


def axes_flat(axes):
    return np.asarray(axes).reshape(-1)


def plane_angles(eq, nplanes):
    """Toroidal angles of the section/plot planes.

    Matches DESC's poincare_plot / plot_surfaces integer convention, which
    spaces planes over a single field period ``[0, 2*pi/NFP)`` and folds the
    Poincare punctures into that range. For NFP=1 this is the full torus, so
    HBT behavior is unchanged; for NFP>1 it is what makes the LCFS overlay,
    the puncture labels, and the inside-plasma filter line up with the data.
    """
    return np.linspace(0.0, 2.0 * np.pi / eq.NFP, int(nplanes), endpoint=False)


def overlay_lcfs(rt, eq, axes, phi_values, theta=128):
    for i, phi in enumerate(np.atleast_1d(phi_values)):
        if i >= len(axes):
            break
        grid = rt.LinearGrid(
            rho=np.array([1.0]),
            theta=theta,
            zeta=np.array([phi]),
            # phi is a physical toroidal angle over the full torus; the eq is
            # periodic, so NFP=1 evaluates geometry correctly and avoids DESC's
            # "LinearGrid should be defined on 1 field period" restriction (NFP>1).
            NFP=1,
        )
        data = eq.compute(["R", "Z"], grid=grid)
        axes[i].plot(
            np.asarray(data["R"], dtype=float),
            np.asarray(data["Z"], dtype=float),
            color="tab:orange",
            linewidth=1.0,
        )


def plot_poincare(rt, eq, field, field_lines, args, out_path):
    planes, crossings = poincare_crossings(field_lines, args.phi_planes)
    fig, axes = rt.plot_surfaces(eq, theta=0, phi=planes, rho=np.array([1.0]))
    axes = axes_flat(axes)

    colors = rt.plt.cm.viridis(np.linspace(0.0, 1.0, field_lines.shape[0]))
    for plane_idx, points in enumerate(crossings):
        ax = axes[plane_idx]
        if points:
            arr = np.asarray(points)
            seed_idx = arr[:, 2].astype(int)
            ax.scatter(
                arr[:, 0],
                arr[:, 1],
                s=args.marker_size,
                color=colors[seed_idx],
                alpha=0.85,
                linewidths=0.0,
            )
        ax.set_xlim(*args.bounds_R)
        ax.set_ylim(*args.bounds_Z)
        ax.set_aspect("equal")
        ax.set_title(f"phi = {planes[plane_idx]:.3f}")

    fig.savefig(out_path, dpi=500)
    rt.plt.close(fig)
    print(f"Saved Poincare plot to {out_path}")


def plot_poincare_data(rt, eq, field, data, args, out_path):
    R, Z = normalize_poincare_data(data, args)
    nplanes = R.shape[1]
    nseeds = R.shape[2]

    fig, axes = rt.plt.subplots(
        2 if nplanes > 3 else 1,
        int(np.ceil(nplanes / (2 if nplanes > 3 else 1))),
        squeeze=False,
        figsize=(12, 8 if nplanes > 3 else 4),
    )
    axes = axes_flat(axes)
    colors = rt.plt.cm.viridis(np.linspace(0.0, 1.0, max(nseeds, 1)))
    planes = plane_angles(eq, nplanes)

    for plane_idx in range(nplanes):
        ax = axes[plane_idx]
        for seed_idx in range(nseeds):
            finite = np.isfinite(R[:, plane_idx, seed_idx]) & np.isfinite(
                Z[:, plane_idx, seed_idx]
            )
            if np.any(finite):
                ax.scatter(
                    R[finite, plane_idx, seed_idx],
                    Z[finite, plane_idx, seed_idx],
                    s=args.marker_size,
                    color=colors[seed_idx],
                    alpha=0.85,
                    linewidths=0.0,
                )
        ax.set_xlim(*args.bounds_R)
        ax.set_ylim(*args.bounds_Z)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R$ (m)")
        ax.set_ylabel(r"$Z$ (m)")
        ax.set_title(f"phi = {planes[plane_idx]:.3f}")

    for ax in axes[nplanes:]:
        ax.set_visible(False)

    overlay_lcfs(rt, eq, axes, planes, theta=args.inside_check_theta)
    fig.savefig(out_path, dpi=500)
    rt.plt.close(fig)
    print(f"Saved Poincare plot to {out_path}")


def run_poincare_plot(rt, eq, field, plasma_modes, box, shift, seeds_world, args, out_path):
    if abs(args.seed_phi) > 1e-14:
        print(
            "warning: DESC poincare_plot takes R/Z seeds on its initial plane; "
            f"seed_phi={args.seed_phi} is only used to generate seed R/Z values."
        )

    trace_field = make_external_plus_nufft_field(
        rt,
        field,
        plasma_modes,
        box,
        shift,
        args,
    )
    seeds_rpz = xyz_to_rpz(seeds_world)
    print("Tracing with DESC poincare_plot using ExternalPlusNUFFTField...")
    fig, axes, data = rt.poincare_plot(
        trace_field,
        seeds_rpz[:, 0],
        seeds_rpz[:, 2],
        ntransit=args.ntransit,
        phi=args.phi_planes,
        NFP=eq.NFP,
        return_data=True,
        size=args.marker_size,
        color="red",
        bounds_R=tuple(args.bounds_R),
        bounds_Z=tuple(args.bounds_Z),
        chunk_size=args.field_chunk_size,
        bs_chunk_size=args.field_chunk_size,
        max_steps=args.poincare_max_steps,
        rtol=args.poincare_rtol,
        atol=args.poincare_atol,
        min_step_size=args.poincare_min_step_size,
    )
    rt.plt.close(fig)

    data, seeds_world = filter_poincare_data_outside_plasma(
        rt,
        eq,
        data,
        seeds_world,
        args,
    )
    plot_poincare_data(rt, eq, field, data, args, out_path)
    return data, seeds_world


def evaluate_quiver_field(rt, field, plasma_modes, coords_world, box, shift, args):
    targets_local = np.asarray(coords_world, dtype=float) - shift
    bx_p, by_p, bz_p = rt.eval_B(
        plasma_modes[0],
        plasma_modes[1],
        plasma_modes[2],
        rt.jnp.asarray(targets_local),
        box,
        eps=args.nufft_eps,
    )
    B_plasma = np.stack(
        [np.asarray(bx_p), np.asarray(by_p), np.asarray(bz_p)],
        axis=1,
    )
    if args.plasma_only:
        return B_plasma

    B_external = np.asarray(
        field.compute_magnetic_field(
            coords_world,
            basis="xyz",
            chunk_size=args.field_chunk_size,
        )
    )
    return B_plasma + B_external


def plot_quiver(rt, eq, field, plasma_modes, box, shift, args, out_path):
    phi_values = plane_angles(eq, args.quiver_planes)
    offsets = np.linspace(
        args.quiver_offset_min,
        args.quiver_offset_max,
        args.quiver_offset_count,
    )

    # Pass the explicit angles (not the integer count) so plot_surfaces draws
    # the LCFS at the same planes where we evaluate the quiver arrows.
    fig, axes = rt.plot_surfaces(eq, theta=0, phi=phi_values, rho=np.array([1.0]))
    axes = axes_flat(axes)

    for i, phi in enumerate(phi_values):
        ax = axes[i]
        grid = rt.LinearGrid(
            rho=np.array([1.0]),
            theta=args.quiver_theta,
            zeta=np.array([phi]),
            # phi is a physical toroidal angle over the full torus; the eq is
            # periodic, so NFP=1 evaluates geometry correctly and avoids DESC's
            # "LinearGrid should be defined on 1 field period" restriction (NFP>1).
            NFP=1,
        )
        data = eq.compute(["x", "n_rho"], grid=grid, basis="xyz")
        surface = np.asarray(data["x"], dtype=float)
        normal = np.asarray(data["n_rho"], dtype=float)
        coords_world = np.concatenate([surface + offset * normal for offset in offsets], axis=0)
        coords_rpz = xyz_to_rpz(coords_world)

        B_xyz = evaluate_quiver_field(
            rt,
            field,
            plasma_modes,
            coords_world,
            box,
            shift,
            args,
        )
        B_rpz = xyz_vec_to_rpz(B_xyz, coords_rpz[:, 1])
        ax.quiver(
            coords_rpz[:, 0],
            coords_rpz[:, 2],
            B_rpz[:, 0],
            B_rpz[:, 2],
            angles="xy",
        )
        ax.set_xlim(args.quiver_R[0] - 0.05, args.quiver_R[1] + 0.05)
        ax.set_ylim(args.quiver_Z[0] - 0.05, args.quiver_Z[1] + 0.05)
        ax.set_aspect("equal")
        ax.set_title(f"phi = {phi:.3f}")

    fig.savefig(out_path, dpi=500)
    rt.plt.close(fig)
    print(f"Saved quiver plot to {out_path}")


def save_trace_data(path, field_lines, seeds_world, shift, box):
    np.savez_compressed(
        path,
        field_lines=field_lines,
        seeds_world=seeds_world,
        shift=shift,
        box_lengths=np.array([box.Lx, box.Ly, box.Lz], dtype=float),
        box_cells=np.array([box.Nx, box.Ny, box.Nz], dtype=int),
    )
    print(f"Saved trace data to {path}")


def save_poincare_data(path, data, seeds_world, shift, box):
    arrays = {
        f"poincare_{key}": np.asarray(value)
        for key, value in data.items()
        if np.asarray(value).dtype != object
    }
    np.savez_compressed(
        path,
        **arrays,
        seeds_world=seeds_world,
        shift=shift,
        box_lengths=np.array([box.Lx, box.Ly, box.Lz], dtype=float),
        box_cells=np.array([box.Nx, box.Ny, box.Nz], dtype=int),
    )
    print(f"Saved Poincare data to {path}")


def lcfs_bounding_box(rt, eq, n_phi=64, n_theta=256):
    """(R_min, R_max, Z_min, Z_max) of the LCFS over all toroidal planes.

    Sampled directly from the loaded equilibrium so plot bounds adapt to any
    config's geometry instead of assuming the HBT cross-section.
    """
    phi_values = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    R_all = []
    Z_all = []
    for phi in phi_values:
        grid = rt.LinearGrid(
            rho=np.array([1.0]),
            theta=n_theta,
            zeta=np.array([phi]),
            # phi is a physical toroidal angle over the full torus; the eq is
            # periodic, so NFP=1 evaluates geometry correctly and avoids DESC's
            # "LinearGrid should be defined on 1 field period" restriction (NFP>1).
            NFP=1,
        )
        data = eq.compute(["R", "Z"], grid=grid)
        R_all.append(np.asarray(data["R"], dtype=float))
        Z_all.append(np.asarray(data["Z"], dtype=float))
    R_all = np.concatenate(R_all)
    Z_all = np.concatenate(Z_all)
    return (
        float(R_all.min()),
        float(R_all.max()),
        float(Z_all.min()),
        float(Z_all.max()),
    )


def resolve_plot_bounds(rt, eq, args):
    """Fill any unset plot/quiver bounds from the LCFS bounding box.

    The Poincare window is padded by --bounds-margin (default: enough to show
    the exterior seeds out to --seed-offset-max plus breathing room); the quiver
    window is padded by --quiver-offset-max. Explicit --bounds-* are untouched.
    """
    need_poincare = args.bounds_R is None or args.bounds_Z is None
    need_quiver = (not args.skip_quiver) and (
        args.quiver_R is None or args.quiver_Z is None
    )
    if not (need_poincare or need_quiver):
        return

    R_min, R_max, Z_min, Z_max = lcfs_bounding_box(rt, eq)
    print(
        f"LCFS bounding box: R=[{R_min:.4f}, {R_max:.4f}] m, "
        f"Z=[{Z_min:.4f}, {Z_max:.4f}] m"
    )

    if need_poincare:
        if args.bounds_margin is not None:
            margin = args.bounds_margin
        else:
            span = max(R_max - R_min, Z_max - Z_min)
            margin = max(args.seed_offset_max, 0.1 * span)
        if args.bounds_R is None:
            args.bounds_R = (R_min - margin, R_max + margin)
        if args.bounds_Z is None:
            args.bounds_Z = (Z_min - margin, Z_max + margin)
        print(
            f"Auto Poincare bounds (margin {margin:.4f} m): "
            f"R={tuple(round(v, 4) for v in args.bounds_R)}, "
            f"Z={tuple(round(v, 4) for v in args.bounds_Z)}"
        )

    if need_quiver:
        qmargin = args.quiver_offset_max
        if args.quiver_R is None:
            args.quiver_R = (R_min - qmargin, R_max + qmargin)
        if args.quiver_Z is None:
            args.quiver_Z = (Z_min - qmargin, Z_max + qmargin)
        print(
            f"Auto quiver bounds (margin {qmargin:.4f} m): "
            f"R={tuple(round(v, 4) for v in args.quiver_R)}, "
            f"Z={tuple(round(v, 4) for v in args.quiver_Z)}"
        )


def main():
    args = parse_args()
    apply_config(args)
    apply_cluster(args)
    if not args.run and not args.local:
        submit_slurm_job(args)
        return

    configure_paths()
    rt = import_runtime(args)
    name, _, save_dir, eq, field = load_case(rt, args)
    resolve_plot_bounds(rt, eq, args)

    box, shift, plasma_modes, Bx_p, By_p, Bz_p, _ = build_plasma_field_grid(rt, eq, args)
    seeds_world = make_surface_offset_seeds(rt, eq, args)
    poincare_data = None
    field_lines = None

    if not args.skip_poincare:
        if args.trace_method == "poincare":
            poincare_data, seeds_world = run_poincare_plot(
                rt,
                eq,
                field,
                plasma_modes,
                box,
                shift,
                seeds_world,
                args,
                save_dir / f"fieldlines-nufft-{name}.png",
            )
        else:
            if args.trace_external_mode == "grid":
                Bx, By, Bz = add_external_field_grid(
                    rt,
                    field,
                    box,
                    shift,
                    Bx_p,
                    By_p,
                    Bz_p,
                    args,
                )
            else:
                print("Skipping external-field grid for tracing (--trace-external-mode direct).")
                Bx, By, Bz = np.asarray(Bx_p), np.asarray(By_p), np.asarray(Bz_p)

            field_lines = trace_lines(rt, field, Bx, By, Bz, box, shift, seeds_world, args)
            field_lines, seeds_world = filter_lines_outside_plasma(
                rt,
                eq,
                field_lines,
                seeds_world,
                args,
            )
            plot_poincare(
                rt,
                eq,
                field,
                field_lines,
                args,
                save_dir / f"fieldlines-nufft-{name}.png",
            )

    if not args.skip_quiver:
        plot_quiver(
            rt,
            eq,
            field,
            plasma_modes,
            box,
            shift,
            args,
            save_dir / f"bfield-nufft-{name}.png",
        )

    if args.save_data and poincare_data is not None:
        save_poincare_data(
            save_dir / f"fieldlines-nufft-{name}.npz",
            poincare_data,
            seeds_world,
            shift,
            box,
        )
    elif args.save_data and field_lines is not None:
        save_trace_data(
            save_dir / f"fieldlines-nufft-{name}.npz",
            field_lines,
            seeds_world,
            shift,
            box,
        )


if __name__ == "__main__":
    main()
