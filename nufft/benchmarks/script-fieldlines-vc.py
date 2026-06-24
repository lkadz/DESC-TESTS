"""Trace field lines with a DESC virtual-casing Biot-Savart plasma field.

This is the virtual-casing analogue of ``script-fieldlines-nufft.py``: same CLI,
seed generation, Poincare/quiver plots, inside-plasma filter, auto plot bounds,
variant folders, and Slurm submission -- but the plasma field is computed by the
surface-integral virtual-casing principle (as in ``fieldline-tracing.ipynb``)
instead of the NUFFT volume Biot-Savart.

The plasma field is evaluated by integrating the ``_kernel_biot_savart`` surface
kernel over the LCFS (DESC's virtual-casing method), wrapped together with the
optimized external coils in an ``ExternalPlusVirtualCasingField``. DESC's
``poincare_plot`` traces that total field.

Example (mirrors the NUFFT runs):

    python benchmarks/script-fieldlines-vc.py \\
      --config precise_QA \\
      --time 00:59:00 \\
      --seed-poloidal 10 \\
      --seed-offset-min 0.02 \\
      --seed-offset-max 0.08 \\
      --seed-offset-count 4 \\
      --ntransit 30 \\
      --poincare-rtol 1e-5 --poincare-atol 1e-5

On a Slurm cluster the command above submits a job that re-runs this script
with ``--run`` to do the tracing on a compute node.
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
    for path in (SCRIPT_DIR, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace field lines using DESC virtual-casing surface Biot-Savart for "
            "the plasma field, plus the optimized external coils."
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
            "Auto-fills --name, --eq-file (eq.h5), and --coil-file (coils.h5)."
        ),
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        help=(
            "Run variant tag. When --config is set, results go to "
            "results/desc_bfield/<config>/<variant>/field_line_tracing_vc/. "
            "Defaults to 'baseline' (or vcM<M>_N<N> for non-default resolution)."
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
    parser.add_argument("--slurm-mem", default="64G", help="Slurm memory request.")
    parser.add_argument("--slurm-cores", type=int, default=8, help="Slurm CPU-core request.")
    parser.add_argument("--slurm-gres", default="gpu:1", help="Slurm generic resource request. Use '' to omit.")
    parser.add_argument(
        "--cluster",
        default=None,
        choices=("della", "della40", "adroit", "stellar"),
        help=(
            "Convenience preset for Slurm partition + constraint: "
            "della -> ('gpu', 'nomig'); della40 -> ('gpu', 'nomig&gpu40'); "
            "adroit -> ('gpu', 'gpu80'); stellar -> ('gpu', ''). "
            "Explicit --slurm-partition/--slurm-constraint override the preset."
        ),
    )
    parser.add_argument("--slurm-partition", default=None, help="Slurm partition. Defaults per --cluster (else empty).")
    parser.add_argument("--slurm-constraint", default=None, help="Slurm node constraint. Defaults per --cluster (else empty).")
    parser.add_argument("--conda-env", default="desc-env", help="Conda environment activated inside the Slurm job.")
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
        help="Optional explicit equilibrium h5 file (e.g. a config's eq.h5).",
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
    parser.add_argument(
        "--field-chunk-size",
        type=int,
        default=50000,
        help=(
            "Evaluation points per chunk for the virtual-casing surface integral "
            "and the external-coil Biot-Savart (passed to poincare_plot)."
        ),
    )
    parser.add_argument(
        "--plasma-only",
        action="store_true",
        help="Trace only the virtual-casing plasma field; do not add external coils.",
    )

    # Virtual-casing source-surface resolution (LinearGrid M, N on the LCFS).
    parser.add_argument(
        "--vc-M",
        type=int,
        default=256,
        help="Poloidal resolution of the LCFS virtual-casing source grid.",
    )
    parser.add_argument(
        "--vc-N",
        type=int,
        default=256,
        help="Toroidal resolution of the LCFS virtual-casing source grid.",
    )

    parser.add_argument("--seed-theta", "--seed-poloidal", dest="seed_theta", type=int, default=10, help="Number of poloidal seed points on the LCFS offset surface.")
    parser.add_argument("--seed-phi", type=float, default=0.0)
    parser.add_argument("--seed-offset-min", type=float, default=0.01)
    parser.add_argument("--seed-offset-max", type=float, default=0.03)
    parser.add_argument("--seed-offset-count", type=int, default=3)

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
        help="Poincare plot R window. Default: auto from the LCFS bounding box.",
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
        help="Padding (m) added on every side of the LCFS bounding box for auto bounds.",
    )
    parser.add_argument("--marker-size", type=float, default=0.5)
    parser.add_argument("--skip-poincare", action="store_true")
    parser.add_argument(
        "--include-inside-plasma",
        action="store_true",
        help="Keep traced seeds even if they enter the plasma. Default is to exclude them.",
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
        help="Also save Poincare data to an npz file.",
    )
    return parser.parse_args()


def auto_variant(args):
    """Build a variant tag from VC resolution if --variant was not set."""
    if args.variant != "baseline":
        return args.variant
    if args.vc_M == 256 and args.vc_N == 256:
        return "baseline"
    return f"vcM{args.vc_M}_N{args.vc_N}"


def output_dir(args):
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
    elif args.config is not None:
        variant = auto_variant(args)
        save_dir = REPO_ROOT / "results" / "desc_bfield" / args.config / variant / "field_line_tracing_vc"
    elif args.local:
        save_dir = SCRIPT_DIR / "results-fieldlines-vc-local"
    else:
        save_dir = Path("/scratch/gpfs/EKOLEMEN/hbt-ep/results-fieldlines-vc")
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
        "--field-chunk-size",
        str(args.field_chunk_size),
        "--vc-M",
        str(args.vc_M),
        "--vc-N",
        str(args.vc_N),
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
    """Resolve --config into --name/--eq-file/--coil-file via bench_config."""
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
    # Della's default partition is 'cpu'; GPUs live in 'gpu'. 'nomig' excludes
    # MIG slices (1g.10gb, 3g.40gb), which DESC cannot use. della40 pins the
    # full 40GB A100 (shorter queue); gpu40 alone also matches the 3g.40gb MIG
    # slice, so it must be ANDed with nomig.
    "della": {"partition": "gpu", "constraint": "nomig"},
    "della40": {"partition": "gpu", "constraint": "nomig&gpu40"},
    "adroit": {"partition": "gpu", "constraint": "gpu80"},
    "stellar": {"partition": "gpu", "constraint": ""},
}


def apply_cluster(args):
    """Fill Slurm partition/constraint from --cluster unless set explicitly.

    della  -> full non-MIG A100 via the 'nomig' feature in the default partition.
    adroit -> full A100 80GB node (feature 'gpu80') in the 'gpu' partition,
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
        return f"fl_vc_{safe}"
    n0 = f"{args.bump_n0:g}".replace("-", "m").replace(".", "p")
    n1 = f"{args.bump_n1:g}".replace("-", "m").replace(".", "p")
    ki = f"{args.k_iota:g}".replace("-", "m").replace(".", "p")
    return f"fl_vc_n0_{n0}_n1_{n1}_ki_{ki}"


def submit_slurm_job(args):
    save_dir = output_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    job_name = slurm_job_name(args)
    slurm_path = SCRIPT_DIR / "job.slurm_vc_fieldlines"
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

    # Let DESC choose the device before anything asks JAX to enumerate CUDA.
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

    from desc.backend import print_backend_info, fori_loop
    from desc.batching import batch_map
    from desc.grid import LinearGrid
    from desc.io import load
    from desc.utils import errorif, xyz2rpz, xyz2rpz_vec, rpz2xyz_vec
    from desc.integrals.singularities import _kernel_biot_savart
    from desc.compute.utils import _compute as compute_fun
    from desc.compute.utils import get_transforms, get_profiles
    from desc.magnetic_fields._core import _MagneticField
    from desc.plotting import poincare_plot, plot_surfaces

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
        fori_loop=fori_loop,
        batch_map=batch_map,
        errorif=errorif,
        xyz2rpz=xyz2rpz,
        xyz2rpz_vec=xyz2rpz_vec,
        rpz2xyz_vec=rpz2xyz_vec,
        _kernel_biot_savart=_kernel_biot_savart,
        compute_fun=compute_fun,
        get_transforms=get_transforms,
        get_profiles=get_profiles,
        gpu_device=gpu_device,
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

    print(f"Loading equilibrium: {eq_path}")
    eq = rt.load(eq_path)
    # eq.h5 may hold the full solve history (EquilibriaFamily); take the final
    # converged equilibrium.
    if eq.__class__.__name__ == "EquilibriaFamily":
        print(f"  loaded EquilibriaFamily with {len(eq)} states; using the last.")
        eq = eq[-1]

    field = None
    if not args.plasma_only:
        if not coil_path.exists():
            raise FileNotFoundError(f"Could not find {coil_path}")
        print(f"Loading external field: {coil_path}")
        field = rt.load(coil_path)
    else:
        print("Skipping external field (--plasma-only).")
    return name, source_dir, save_dir, eq, field


def xyz_to_rpz(xyz):
    xyz = np.asarray(xyz, dtype=float)
    out = np.empty_like(xyz)
    out[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    out[:, 1] = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    out[:, 2] = xyz[:, 2]
    return out


# ---------------------------------------------------------------------------
# Virtual-casing plasma field (ported from fieldline-tracing.ipynb)
# ---------------------------------------------------------------------------
def _vc_integrate_surface(rt, coords, source_data, source_grid, kernel, chunk_size=None):
    """Integrate the Biot-Savart kernel over the LCFS at points outside it."""
    jnp = rt.jnp
    assert source_grid.num_rho == 1, (
        "source_grid must be on a flux surface. "
        f"Got source_grid.num_rho = {source_grid.num_rho}"
    )

    source_zeta = source_data.setdefault("zeta", source_grid.nodes[:, 2])
    source_phi = source_data["phi"]

    eval_data = {"R": coords[:, 0], "phi": coords[:, 1], "Z": coords[:, 2]}

    ht = 2 * jnp.pi / source_grid.num_theta
    hz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
    w = source_data["|e_theta x e_zeta|"][jnp.newaxis] * ht * hz

    def nfp_loop(j, f_data):
        """Calculate effects from source points on a single field period."""
        f, source_data = f_data
        source_data["zeta"] = (source_zeta + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )
        source_data["phi"] = (source_phi + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )

        def eval_pt(eval_data_i):
            k = kernel(eval_data_i, source_data).reshape(
                -1, source_grid.num_nodes, kernel.ndim
            )
            return jnp.sum(k * w[..., jnp.newaxis], axis=1)

        f += rt.batch_map(eval_pt, eval_data, chunk_size).reshape(
            coords.shape[0], kernel.ndim
        )
        return f, source_data

    rt.errorif(
        source_grid.num_zeta == 1 and source_grid.NFP == 1,
        msg="Source grid cannot compute toroidal effects.\n"
        "Increase NFP of source grid to e.g. 64.",
    )
    f = jnp.zeros((coords.shape[0], kernel.ndim))
    f, _ = rt.fori_loop(0, source_grid.NFP, nfp_loop, (f, source_data))

    source_data["zeta"] = source_zeta
    source_data["phi"] = source_phi

    if kernel.ndim == 3:
        f = rt.xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def _vc_eq_magnetic_field(
    rt,
    eq,
    coords,
    params=None,
    basis="rpz",
    source_grid=None,
    transforms=None,
    profiles=None,
    chunk_size=None,
):
    jnp = rt.jnp
    coords = jnp.atleast_2d(coords)
    eval_rpz = rt.xyz2rpz(coords) if basis.lower() == "xyz" else coords

    if source_grid is None:
        source_grid = rt.LinearGrid(
            rho=jnp.array([1.0]),
            M=256,
            N=256,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )

    kernel = rt._kernel_biot_savart
    data = rt.compute_fun(
        eq,
        kernel.keys,
        grid=source_grid,
        params=params,
        transforms=transforms,
        profiles=profiles,
    )

    B = _vc_integrate_surface(rt, eval_rpz, data, source_grid, kernel, chunk_size=chunk_size)

    if basis.lower() == "xyz":
        B = rt.rpz2xyz_vec(B, phi=coords[:, 1])
    return B


def make_vc_source_grid(rt, eq, args):
    nfp = int(eq.NFP) if eq.N > 0 else 64
    return rt.LinearGrid(
        rho=np.array([1.0]),
        M=args.vc_M,
        N=args.vc_N,
        NFP=nfp,
        sym=False,
    )


def make_external_plus_vc_field(rt, eq, ext_field, source_grid_vc, args):
    base_static = list(getattr(rt.MagneticFieldBase, "_static_attrs", []))
    kernel = rt._kernel_biot_savart

    class ExternalPlusVirtualCasingField(rt.MagneticFieldBase):
        """External coil field + virtual-casing plasma field."""

        _static_attrs = base_static + ["source_grid_vc", "plasma_only"]

        def __init__(self):
            self.eq = eq
            self.ext_field = ext_field
            self.source_grid_vc = source_grid_vc
            self.plasma_only = args.plasma_only
            # Build constant transform/profile matrices once at construction.
            self.vc_transforms = rt.get_transforms(kernel.keys, eq, source_grid_vc)
            self.vc_profiles = rt.get_profiles(kernel.keys, eq, source_grid_vc)

        def compute_magnetic_field(
            self,
            coords,
            params=None,
            basis="rpz",
            source_grid=None,
            transforms=None,
            chunk_size=None,
        ):
            if params is None:
                params = self.eq.params_dict

            B_plasma = _vc_eq_magnetic_field(
                rt,
                self.eq,
                coords,
                params=params,
                basis=basis,
                source_grid=self.source_grid_vc,
                transforms=self.vc_transforms,
                profiles=self.vc_profiles,
                chunk_size=chunk_size,
            )

            if self.plasma_only:
                return B_plasma

            B_ext = self.ext_field.compute_magnetic_field(
                coords,
                basis=basis,
                chunk_size=chunk_size,
            )
            return B_ext + B_plasma

        def compute_magnetic_vector_potential(self, *args, **kwargs):
            raise NotImplementedError

    return ExternalPlusVirtualCasingField()


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
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
        NFP=1,
    )
    data = eq.compute(["x", "n_rho"], grid=grid, basis="xyz")
    surface = np.asarray(data["x"], dtype=float)
    normal = np.asarray(data["n_rho"], dtype=float)

    seeds = [surface + offset * normal for offset in offsets]
    seeds = np.concatenate(seeds, axis=0)
    print(f"Generated {len(seeds)} seeds from {len(offsets)} LCFS offsets.")
    return seeds


# ---------------------------------------------------------------------------
# Inside-plasma filter and plotting helpers (shared with the NUFFT script)
# ---------------------------------------------------------------------------
def axes_flat(axes):
    return np.asarray(axes).reshape(-1)


def plane_angles(eq, nplanes):
    """Toroidal angles of the section/plot planes (DESC integer convention)."""
    return np.linspace(0.0, 2.0 * np.pi / eq.NFP, int(nplanes), endpoint=False)


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

    # Map each plane to the LCFS polygon at the SAME physical phi.
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


def overlay_lcfs(rt, eq, axes, phi_values, theta=128):
    for i, phi in enumerate(np.atleast_1d(phi_values)):
        if i >= len(axes):
            break
        grid = rt.LinearGrid(
            rho=np.array([1.0]),
            theta=theta,
            zeta=np.array([phi]),
            NFP=1,
        )
        data = eq.compute(["R", "Z"], grid=grid)
        axes[i].plot(
            np.asarray(data["R"], dtype=float),
            np.asarray(data["Z"], dtype=float),
            color="tab:orange",
            linewidth=1.0,
        )


def plot_poincare_data(rt, eq, data, args, out_path):
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


def run_poincare_plot(rt, eq, vc_field, seeds_world, args, out_path):
    if abs(args.seed_phi) > 1e-14:
        print(
            "warning: DESC poincare_plot takes R/Z seeds on its initial plane; "
            f"seed_phi={args.seed_phi} is only used to generate seed R/Z values."
        )

    seeds_rpz = xyz_to_rpz(seeds_world)
    print("Tracing with DESC poincare_plot using ExternalPlusVirtualCasingField...")
    fig, axes, data = rt.poincare_plot(
        vc_field,
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

    data, seeds_world = filter_poincare_data_outside_plasma(rt, eq, data, seeds_world, args)
    plot_poincare_data(rt, eq, data, args, out_path)
    return data, seeds_world


def plot_quiver(rt, eq, vc_field, args, out_path):
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
            NFP=1,
        )
        data = eq.compute(["x", "n_rho"], grid=grid, basis="xyz")
        surface = np.asarray(data["x"], dtype=float)
        normal = np.asarray(data["n_rho"], dtype=float)
        coords_world = np.concatenate([surface + offset * normal for offset in offsets], axis=0)
        coords_rpz = xyz_to_rpz(coords_world)

        B_rpz = np.asarray(
            vc_field.compute_magnetic_field(
                coords_rpz,
                basis="rpz",
                chunk_size=args.field_chunk_size,
            )
        )
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


def save_poincare_data(path, data, seeds_world):
    arrays = {
        f"poincare_{key}": np.asarray(value)
        for key, value in data.items()
        if np.asarray(value).dtype != object
    }
    np.savez_compressed(path, **arrays, seeds_world=seeds_world)
    print(f"Saved Poincare data to {path}")


def lcfs_bounding_box(rt, eq, n_phi=64, n_theta=256):
    """(R_min, R_max, Z_min, Z_max) of the LCFS over all toroidal planes."""
    phi_values = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    R_all = []
    Z_all = []
    for phi in phi_values:
        grid = rt.LinearGrid(
            rho=np.array([1.0]),
            theta=n_theta,
            zeta=np.array([phi]),
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
    """Fill any unset plot/quiver bounds from the LCFS bounding box."""
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

    source_grid = make_vc_source_grid(rt, eq, args)
    print(
        f"Virtual-casing source grid: M={args.vc_M}, N={args.vc_N}, "
        f"NFP={source_grid.NFP}, nodes={source_grid.num_nodes}"
    )
    vc_field = make_external_plus_vc_field(rt, eq, field, source_grid, args)

    seeds_world = make_surface_offset_seeds(rt, eq, args)
    poincare_data = None

    if not args.skip_poincare:
        poincare_data, seeds_world = run_poincare_plot(
            rt,
            eq,
            vc_field,
            seeds_world,
            args,
            save_dir / f"fieldlines-vc-{name}.png",
        )

    if not args.skip_quiver:
        plot_quiver(rt, eq, vc_field, args, save_dir / f"bfield-vc-{name}.png")

    if args.save_data and poincare_data is not None:
        save_poincare_data(
            save_dir / f"fieldlines-vc-{name}.npz",
            poincare_data,
            seeds_world,
        )


if __name__ == "__main__":
    main()
