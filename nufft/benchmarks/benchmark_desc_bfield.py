from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from bench_config import available_configs, config_overrides, load_config


CSV_FIELDNAMES = [
    "N",
    "status",
    "compute_B_hat_s",
    "eval_B_s",
    "abs_l2",
    "rel_l2",
    "abs_rms",
    "rel_rms",
    "abs_max",
    "rel_max",
    "mag_rel_rms",
    "Bx_rel_rms",
    "By_rel_rms",
    "Bz_rel_rms",
    "error",
]

SOURCE_SCAN_FIELDNAMES = [
    "n_rho",
    "n_theta",
    "n_zeta",
    "source_points",
    "status",
    "compute_B_hat_s",
    "eval_B_s",
    "rel_l2",
    "rel_rms",
    "mag_rel_rms",
    "rel_max",
    "error",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NUFFT Biot-Savart B against a DESC stellarator "
            "inside the plasma."
        )
    )
    parser.add_argument(
        "--config",
        default="precise_QA",
        help=(
            "Named benchmark configuration (a file in benchmarks/configurations/) "
            "or a path to a .json config. Supplies defaults for the equilibrium, "
            "coils, source model, and grid. Explicit flags below override it. "
            f"Available: {', '.join(available_configs()) or '(none)'}."
        ),
    )
    parser.add_argument(
        "--desc-root",
        default="auto",
        help=(
            "'auto' to use the sibling DESC checkout when present, 'installed' "
            "to use the active environment, or a path to a DESC repo root."
        ),
    )
    parser.add_argument(
        "--eq-file",
        type=Path,
        default=None,
        help=(
            "Override the equilibrium .h5 from the config. "
            "Defaults to benchmarks/configurations/<config>/eq.h5."
        ),
    )
    parser.add_argument(
        "--coil-file",
        type=Path,
        default=None,
        help=(
            "Override the coil/external field .h5 from the config. "
            "Defaults to benchmarks/configurations/<config>/coils.h5."
        ),
    )
    parser.add_argument(
        "--coil-chunk-size",
        type=int,
        default=None,
        help="Chunk size for direct coil/external field evaluation.",
    )
    parser.add_argument(
        "--allow-missing-coils",
        action="store_true",
        help=(
            "Allow a plasma-source-only diagnostic if no paired coil/external "
            "field is available. By default a coil field is required."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/desc_bfield"),
        help="Directory for CSV/JSON/NPZ/PNG outputs.",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Cubic Fourier grid sizes Nx=Ny=Nz to scan.",
    )
    parser.add_argument("--padding", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--n-rho", type=int, default=16)
    parser.add_argument("--n-theta", type=int, default=32)
    parser.add_argument("--n-zeta", type=int, default=64)
    parser.add_argument(
        "--target-rho-max",
        type=float,
        default=1.0,
        help=(
            "Outermost flux-surface radius of the B evaluation grid. 1.0 reaches "
            "the LCFS; use e.g. 0.95 to pull targets off the current "
            "discontinuity at the plasma boundary (diagnostic 1)."
        ),
    )
    parser.add_argument(
        "--edge-taper-rho0",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.95), smoothly taper the volume current to zero "
            "between this rho and the LCFS, removing the current jump (and its "
            "Gibbs ringing). NOTE: this modifies the source, so B becomes the "
            "field of a reduced edge current, not the true DESC current."
        ),
    )
    parser.add_argument(
        "--edge-taper-shape",
        choices=("smoothstep", "smootherstep", "cosine", "quadratic"),
        default="smoothstep",
        help="Edge taper window shape (see desc_interface.edge_taper).",
    )
    parser.add_argument(
        "--spectral-filter",
        choices=("none", "exponential", "lanczos", "cesaro", "raised_cosine"),
        default="none",
        help=(
            "Filter the computed field coefficients B_hat to suppress the Gibbs "
            "ringing from the current discontinuity. Unlike --edge-taper this "
            "does NOT modify the source current, so it adds no current bias."
        ),
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=8,
        help=(
            "Order p of the exponential filter exp(-alpha (|k|/k_max)**p). "
            "Higher p = sharper cutoff that preserves more low-|k| modes. "
            "Ignored by the lanczos/cesaro/raised_cosine filters."
        ),
    )
    parser.add_argument(
        "--lcfs-zoom",
        action="store_true",
        help=(
            "Extra diagnostic plot: BR, BZ, Bphi on a dense band of near-edge flux "
            "surfaces vs poloidal angle, DESC vs NUFFT side by side, to reveal any "
            "Gibbs ringing in the spectral field as rho -> 1 (the LCFS current "
            "discontinuity). Uses the plotted box N and source resolution."
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
        help=(
            "Poloidal sampling of the LCFS zoom (high so Gibbs oscillations are "
            "resolved; the main target grid is too coarse)."
        ),
    )
    parser.add_argument(
        "--source-scan",
        action="store_true",
        help=(
            "Diagnostic 2: hold the box grid fixed (--source-scan-n) and scan the "
            "DESC source resolution from --source-grids. This is the correct "
            "convergence axis for the spectral Biot-Savart method."
        ),
    )
    parser.add_argument(
        "--source-grids",
        nargs="+",
        default=["8,16,32", "16,32,64", "32,64,128"],
        help=(
            "Source resolutions for --source-scan as 'n_rho,n_theta,n_zeta' "
            "triples."
        ),
    )
    parser.add_argument(
        "--source-scan-n",
        type=int,
        default=64,
        help="Fixed box grid Nx=Ny=Nz used during --source-scan.",
    )
    parser.add_argument(
        "--joint-scan",
        action="store_true",
        help=(
            "Convergence scan that refines the box grid N and the source "
            "resolution together (matched refinement). This is the convergence "
            "path that actually decreases for the spectral Biot-Savart method."
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
        help=(
            "Matched 'N,n_rho,n_theta,n_zeta' tuples for --joint-scan. The "
            "default holds n_rho=N/4, n_theta=N/2, n_zeta=N."
        ),
    )
    parser.add_argument(
        "--source-model",
        choices=("volume", "boundary"),
        default="volume",
        help=(
            "NUFFT plasma source model. 'volume' uses DESC J. 'boundary' uses "
            "the equivalent LCFS sheet current as a diagnostic."
        ),
    )
    parser.add_argument(
        "--boundary-current-sign",
        type=float,
        default=-1.0,
        help="Multiplier for DESC K_vc in --source-model boundary diagnostics.",
    )
    parser.add_argument(
        "--plot-n",
        type=int,
        default=None,
        help="Grid size to use for quiver/component plots. Defaults to largest success.",
    )
    parser.add_argument(
        "--num-cross-sections",
        type=int,
        default=4,
        help="Number of toroidal cross sections to plot from one field period.",
    )
    parser.add_argument(
        "--max-quiver-arrows",
        type=int,
        default=180,
        help="Maximum arrows per quiver panel.",
    )
    parser.add_argument(
        "--x64",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable JAX x64. Recommended for eps=1e-12.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=("auto", "cpu", "gpu", "cuda", "rocm", "tpu"),
        default="auto",
        help=(
            "JAX platform selector. Use 'cuda' for NVIDIA GPUs. 'gpu' is "
            "accepted as a CUDA alias. Use 'auto' to let JAX choose from the "
            "job environment."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later N values after a catchable failure.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    # eq_file / coil_file from the config dir become defaults; explicit CLI flags win.
    parser.set_defaults(**config_overrides(config))
    args = parser.parse_args()
    args.config_name = config["name"]
    return args


def configure_paths(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))

    if args.desc_root == "installed":
        return repo_root

    desc_root = None
    if args.desc_root == "auto":
        candidate = repo_root.parent
        if (candidate / "desc" / "examples" / "W7-X_output.h5").exists():
            desc_root = candidate
    else:
        desc_root = Path(args.desc_root).expanduser().resolve()

    if desc_root is not None:
        sys.path.insert(0, str(desc_root))
    return repo_root


def configure_jax(args: argparse.Namespace):
    requested_platform = args.jax_platform
    jax_platform = "cuda" if requested_platform == "gpu" else requested_platform

    if jax_platform == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    elif jax_platform in {"cuda", "rocm"}:
        # DESC loads examples under jax.devices("cpu"), so keep CPU visible.
        os.environ["JAX_PLATFORMS"] = f"{jax_platform},cpu"
    else:
        os.environ["JAX_PLATFORMS"] = jax_platform

    from jax import config

    config.update("jax_enable_x64", bool(args.x64))

    import jax
    import jax.numpy as jnp

    try:
        devices = jax.devices()
    except Exception as err:
        raise SystemExit(
            f"Unable to initialize JAX platform {requested_platform!r}: {err}\n"
            "If you are on a login node or an interactive shell without a GPU "
            "allocation, rerun inside an allocated GPU job. For a CPU smoke test, "
            "use `--jax-platform cpu`. To let JAX pick automatically, omit "
            "`--jax-platform` or pass `--jax-platform auto`. For NVIDIA GPUs, "
            "use `--jax-platform cuda`."
        ) from err

    expected_device_platforms = {
        "cuda": {"gpu", "cuda"},
        "gpu": {"gpu", "cuda"},
        "rocm": {"gpu", "rocm"},
    }.get(requested_platform, {requested_platform})
    if requested_platform != "auto" and not any(
        device.platform in expected_device_platforms for device in devices
    ):
        raise SystemExit(
            f"Requested JAX platform {requested_platform!r}, but visible devices are: "
            f"{', '.join(str(device) for device in devices)}"
        )

    return jax, jnp


def load_runtime_modules():
    try:
        import desc
        from desc.grid import LinearGrid
        from desc.io import load as desc_load
        from nufft_biot.desc_interface import (
            desc_equivalent_boundary_current,
            desc_volume_current_on_grid,
        )
        from nufft_biot.field import compute_B_hat, eval_B
        from nufft_biot.types import BoxParams
    except ModuleNotFoundError as err:
        missing = err.name or str(err)
        raise SystemExit(
            "Missing Python dependency while importing DESC/NUFFT modules: "
            f"{missing}\n"
            "Install this repository and its dependencies in the active environment, "
            "for example `python -m pip install -e .`, then rerun. If DESC itself "
            "is missing, also install the DESC requirements."
        ) from err

    return {
        "desc": desc,
        "desc_load": desc_load,
        "LinearGrid": LinearGrid,
        "desc_equivalent_boundary_current": desc_equivalent_boundary_current,
        "desc_volume_current_on_grid": desc_volume_current_on_grid,
        "compute_B_hat": compute_B_hat,
        "eval_B": eval_B,
        "BoxParams": BoxParams,
    }



def block_until_ready(jax, tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def box_geometry(X, Y, Z, padding: float) -> tuple[float, np.ndarray]:
    xmin, xmax = float(np.min(X)), float(np.max(X))
    ymin, ymax = float(np.min(Y)), float(np.max(Y))
    zmin, zmax = float(np.min(Z)), float(np.max(Z))
    side = padding * max(xmax - xmin, ymax - ymin, zmax - zmin)
    center = np.array(
        [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]
    )
    return side, center


def rho_midpoints(n_rho: int) -> np.ndarray:
    return (np.arange(n_rho, dtype=float) + 0.5) / n_rho


def target_rho(n_rho: int, rho_max: float = 1.0) -> np.ndarray:
    """Flux-surface radii for the B evaluation grid.

    The axis (rho=0) is excluded and the outermost shell sits at ``rho_max``.
    With ``rho_max=1`` the grid reaches the LCFS exactly; pulling it inward
    (e.g. 0.95) keeps the targets away from the current discontinuity at the
    plasma boundary, where the spectral Biot-Savart field rings.
    """
    return np.linspace(0.0, rho_max, n_rho + 1)[1:]


def load_equilibrium(args, desc_load):
    if args.eq_file is None:
        raise SystemExit(
            "No equilibrium file found. Copy eq.h5 to "
            f"benchmarks/configurations/{args.config}/ or pass --eq-file."
        )
    eq = desc_load(str(args.eq_file))
    try:
        if eq.__class__.__name__ == "EquilibriaFamily":
            eq = eq[-1]
    except AttributeError:
        pass
    return eq


def load_external_field(args, desc_load):
    if args.coil_file is None:
        return None
    field = desc_load(str(args.coil_file))
    if not hasattr(field, "compute_magnetic_field"):
        raise TypeError(
            f"Loaded coils from {args.coil_file} but it does not provide "
            "compute_magnetic_field."
        )
    return field


def validate_external_field(eq, external_field, args: argparse.Namespace) -> None:
    if external_field is None:
        if not args.allow_missing_coils:
            raise SystemExit(
                f"No coils.h5 found for config {args.config!r}. "
                f"Copy the coil field to benchmarks/configurations/{args.config}/coils.h5, "
                "pass --coil-file, or add --allow-missing-coils for a "
                "plasma-source-only diagnostic."
            )
        return

    field_nfp = getattr(external_field, "NFP", None)
    if field_nfp is None:
        return
    field_nfp = int(field_nfp)
    eq_nfp = int(eq.NFP)
    if field_nfp not in {1, eq_nfp}:
        raise ValueError(
            f"Coil/external field NFP={field_nfp} is incompatible with "
            f"equilibrium NFP={eq_nfp}. Use the matching coil file."
        )


def cylindrical_components(xyz: np.ndarray, B_xyz: np.ndarray) -> tuple[np.ndarray, ...]:
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    B_R = B_xyz[:, 0] * cos_phi + B_xyz[:, 1] * sin_phi
    B_phi = -B_xyz[:, 0] * sin_phi + B_xyz[:, 1] * cos_phi
    B_Z = B_xyz[:, 2]
    return B_R, B_phi, B_Z


def field_metrics(B_desc: np.ndarray, B_nufft: np.ndarray) -> dict[str, float]:
    err = B_nufft - B_desc
    point_err = np.linalg.norm(err, axis=1)
    point_ref = np.linalg.norm(B_desc, axis=1)
    B_desc_norm = np.linalg.norm(B_desc)
    B_desc_rms = np.sqrt(np.mean(point_ref**2))
    mag_desc = point_ref
    mag_nufft = np.linalg.norm(B_nufft, axis=1)

    def safe_div(num, den):
        return float(num / den) if den > 0 else float("nan")

    metrics = {
        "abs_l2": float(np.linalg.norm(err)),
        "rel_l2": safe_div(np.linalg.norm(err), B_desc_norm),
        "abs_rms": float(np.sqrt(np.mean(point_err**2))),
        "rel_rms": safe_div(np.sqrt(np.mean(point_err**2)), B_desc_rms),
        "abs_max": float(np.max(point_err)),
        "rel_max": safe_div(np.max(point_err), np.max(point_ref)),
        "mag_rel_rms": safe_div(
            np.sqrt(np.mean((mag_nufft - mag_desc) ** 2)),
            np.sqrt(np.mean(mag_desc**2)),
        ),
    }
    for i, name in enumerate(("x", "y", "z")):
        metrics[f"B{name}_rel_rms"] = safe_div(
            np.sqrt(np.mean(err[:, i] ** 2)),
            np.sqrt(np.mean(B_desc[:, i] ** 2)),
        )
    return metrics


def append_metrics_csv(path: Path, row: dict[str, float | int | str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_FIELDNAMES})


def append_source_scan_csv(path: Path, row: dict[str, float | int | str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SOURCE_SCAN_FIELDNAMES, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in SOURCE_SCAN_FIELDNAMES})


def parse_source_grids(specs: list[str]) -> list[tuple[int, int, int]]:
    grids = []
    for spec in specs:
        parts = spec.replace(" ", "").split(",")
        if len(parts) != 3:
            raise SystemExit(
                f"Invalid --source-grids entry {spec!r}; expected 'n_rho,n_theta,n_zeta'."
            )
        grids.append(tuple(int(p) for p in parts))
    return grids


def parse_joint_grids(specs: list[str]) -> list[tuple[int, int, int, int]]:
    grids = []
    for spec in specs:
        parts = spec.replace(" ", "").split(",")
        if len(parts) != 4:
            raise SystemExit(
                f"Invalid --joint-grids entry {spec!r}; expected "
                "'N,n_rho,n_theta,n_zeta'."
            )
        grids.append(tuple(int(p) for p in parts))
    return grids


def extract_source(args, modules, eq, n_rho: int, n_theta: int, n_zeta: int):
    """Return (X, Y, Z, Jx, Jy, Jz, w) for the requested source resolution."""
    if args.source_model == "volume":
        return modules["desc_volume_current_on_grid"](
            eq,
            N_rho=n_rho,
            N_theta=n_theta,
            N_zeta=n_zeta,
            replicate_nfp=True,
            taper_rho0=args.edge_taper_rho0,
            taper_shape=args.edge_taper_shape,
        )
    return modules["desc_equivalent_boundary_current"](
        eq,
        N_theta=n_theta,
        N_zeta=n_zeta,
        sign=args.boundary_current_sign,
        replicate_nfp=True,
    )


def run_nufft_once(
    jax,
    jnp,
    modules,
    source,
    box_side: float,
    center: np.ndarray,
    n: int,
    target_xyz: np.ndarray,
    B_external,
    eps: float,
    spectral_filter=None,
    filter_order: int = 8,
):
    """One compute_B_hat + eval_B pass. Returns (B_model, B_plasma, t_hat, t_eval)."""
    X, Y, Z, Jx, Jy, Jz, w = source
    center_jnp = jnp.asarray(center)
    Xb = X - center_jnp[0]
    Yb = Y - center_jnp[1]
    Zb = Z - center_jnp[2]
    target_pos = jnp.asarray(target_xyz) - center_jnp
    box = modules["BoxParams"](box_side, box_side, box_side, n, n, n)

    t0 = time.perf_counter()
    Bx_hat, By_hat, Bz_hat = modules["compute_B_hat"](
        Xb, Yb, Zb, Jx, Jy, Jz, w, box, eps=eps,
        spectral_filter=spectral_filter, filter_order=filter_order,
    )
    block_until_ready(jax, (Bx_hat, By_hat, Bz_hat))
    t_hat = time.perf_counter() - t0

    t1 = time.perf_counter()
    Bx, By, Bz = modules["eval_B"](Bx_hat, By_hat, Bz_hat, target_pos, box, eps=eps)
    B_nufft_jax = jnp.stack([Bx, By, Bz], axis=1)
    block_until_ready(jax, B_nufft_jax)
    t_eval = time.perf_counter() - t1

    B_plasma = np.asarray(B_nufft_jax)
    B_model = B_plasma if B_external is None else B_plasma + B_external
    del box, Bx_hat, By_hat, Bz_hat, Bx, By, Bz, B_nufft_jax
    gc.collect()
    return B_model, B_plasma, t_hat, t_eval


def plot_source_scan(
    rows: list[dict[str, float | int | str]], outdir: Path, configuration_label: str, n: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [row for row in rows if row["status"] == "ok"]
    if len(successful) < 2:
        return

    points = np.array([row["source_points"] for row in successful], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(points, [row["rel_rms"] for row in successful], "s-", label="vector RMS")
    ax.loglog(points, [row["mag_rel_rms"] for row in successful], "^-", label="|B| RMS")
    ax.loglog(points, [row["rel_max"] for row in successful], "D-", label="vector max")
    ax.set_xlabel("source points per field period (n_rho * n_theta * n_zeta)")
    ax.set_ylabel("relative error vs DESC B")
    ax.set_title(f"{configuration_label} source convergence, box N={n}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "error_vs_source.png", dpi=220)
    plt.close(fig)


def select_plot_zetas(nodes: np.ndarray, count: int) -> np.ndarray:
    zetas = np.unique(nodes[:, 2])
    if count >= len(zetas):
        return zetas
    indices = np.linspace(0, len(zetas), count, endpoint=False, dtype=int)
    return zetas[indices]


def flux_surface_lines(eq, LinearGrid, zeta: float, rho_values, theta_count: int = 256):
    lines = []
    for rho in rho_values:
        grid = LinearGrid(
            rho=np.array([rho]),
            theta=theta_count,
            zeta=np.array([zeta]),
            NFP=eq.NFP,
            sym=False,
            axis=False,
        )
        data = eq.compute(["X", "Y", "Z"], grid=grid, basis="xyz")
        xyz = np.column_stack(
            [np.asarray(data["X"]), np.asarray(data["Y"]), np.asarray(data["Z"])]
        )
        R = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        Z = xyz[:, 2]
        lines.append((rho, np.r_[R, R[0]], np.r_[Z, Z[0]]))
    return lines


def compute_outside_quiver(
    plot_data, jax, jnp, modules, eq, LinearGrid,
    source, box_side, center, n, external_field, args,
) -> None:
    """Evaluate the NUFFT model field on a grid extending past the LCFS.

    Stores per-section points outside the plasma boundary plus their model B in
    ``plot_data`` so the quiver plot can show the field beyond the LCFS (where
    DESC has no equilibrium field but Biot-Savart is well-defined).
    """
    from matplotlib.path import Path as MplPath

    nodes = plot_data["target_nodes"]
    zetas = select_plot_zetas(nodes, args.num_cross_sections)

    # Recompute B_hat once at the plotted resolution (B_hat is not retained
    # from the scan loop). Targets outside the source are the accurate regime.
    X, Y, Z, Jx, Jy, Jz, w = source
    cj = jnp.asarray(center)
    box = modules["BoxParams"](box_side, box_side, box_side, n, n, n)
    Bx_hat, By_hat, Bz_hat = modules["compute_B_hat"](
        X - cj[0], Y - cj[1], Z - cj[2], Jx, Jy, Jz, w, box, eps=args.eps,
        spectral_filter=args.spectral_filter, filter_order=args.filter_order,
    )

    all_xyz = []
    all_zeta = []
    for zeta in zetas:
        lcfs = flux_surface_lines(eq, LinearGrid, float(zeta), (1.0,))[0]
        lcfs_R, lcfs_Z = lcfs[1], lcfs[2]
        r0, r1 = float(lcfs_R.min()), float(lcfs_R.max())
        z0, z1 = float(lcfs_Z.min()), float(lcfs_Z.max())
        margin = 0.4 * max(r1 - r0, z1 - z0)
        Rg = np.linspace(r0 - margin, r1 + margin, 22)
        Zg = np.linspace(z0 - margin, z1 + margin, 22)
        RR, ZZ = np.meshgrid(Rg, Zg)
        pts = np.column_stack([RR.ravel(), ZZ.ravel()])
        poly = MplPath(np.column_stack([lcfs_R, lcfs_Z]))
        outside = ~poly.contains_points(pts)
        Rf, Zf = pts[outside, 0], pts[outside, 1]
        phi = float(zeta)
        all_xyz.append(np.column_stack([Rf * np.cos(phi), Rf * np.sin(phi), Zf]))
        all_zeta.append(np.full(Rf.shape[0], phi))

    out_xyz = np.concatenate(all_xyz)
    out_zeta = np.concatenate(all_zeta)

    pos = jnp.asarray(out_xyz) - cj
    Bx, By, Bz = modules["eval_B"](Bx_hat, By_hat, Bz_hat, pos, box, eps=args.eps)
    B_plasma = np.asarray(jnp.stack([Bx, By, Bz], axis=1))
    if external_field is not None:
        B_ext = np.asarray(
            external_field.compute_magnetic_field(
                jnp.asarray(out_xyz), basis="xyz", chunk_size=args.coil_chunk_size
            )
        )
        B_model = B_plasma + B_ext
    else:
        B_model = B_plasma

    plot_data["outside_xyz"] = out_xyz
    plot_data["outside_zeta"] = out_zeta
    plot_data["outside_B"] = B_model


def plot_error_scan(
    rows: list[dict[str, float | int | str]],
    outdir: Path,
    configuration_label: str,
    x_cubed: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [row for row in rows if row["status"] == "ok"]
    if len({row["N"] for row in successful}) < 2:
        # A single N is not a convergence scan. The aggregated plot across all
        # N folders is produced by aggregate_error_scan.py after the jobs run.
        return

    N = np.array([row["N"] for row in successful], dtype=float)
    x = N**3 if x_cubed else N
    xlabel = "N^3 = Nx * Ny * Nz (box points)" if x_cubed else "Nx = Ny = Nz"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(x, [row["rel_rms"] for row in successful], "s-", label="vector RMS")
    ax.loglog(x, [row["mag_rel_rms"] for row in successful], "^-", label="|B| RMS")
    ax.loglog(x, [row["rel_max"] for row in successful], "D-", label="vector max")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative error vs DESC B")
    ax.set_title(f"{configuration_label} DESC B comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "error_vs_N.png", dpi=220)
    plt.close(fig)


def plot_quiver_sections(
    eq,
    LinearGrid,
    plot_data: dict[str, np.ndarray],
    outdir: Path,
    max_arrows: int,
    num_cross_sections: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = plot_data["target_xyz"]
    nodes = plot_data["target_nodes"]
    B_desc = plot_data["B_desc"]
    B_nufft = plot_data["B_nufft"]
    model_label = str(plot_data.get("model_label", "NUFFT"))
    configuration_label = str(plot_data.get("configuration_label", "DESC"))
    N = int(plot_data["N"])
    zetas = select_plot_zetas(nodes, num_cross_sections)
    rho_values = (0.25, 0.5, 0.75, 1.0)

    out_xyz = plot_data.get("outside_xyz")
    out_zeta = plot_data.get("outside_zeta")
    out_B = plot_data.get("outside_B")
    has_outside = out_xyz is not None and len(out_xyz) > 0

    # Smaller arrows than before: larger scale -> shorter unit arrows.
    qscale = 40.0
    qwidth = 0.0032

    def unit(U, V):
        mag = np.sqrt(U**2 + V**2)
        denom = np.where(mag > 0, mag, 1.0)
        return U / denom, V / denom, mag

    for section_index, zeta in enumerate(zetas):
        mask = np.isclose(nodes[:, 2], zeta, rtol=0.0, atol=1e-12)
        xyz_s = xyz[mask]
        R = np.sqrt(xyz_s[:, 0] ** 2 + xyz_s[:, 1] ** 2)
        Z = xyz_s[:, 2]
        BR_desc, _, BZ_desc = cylindrical_components(xyz_s, B_desc[mask])
        BR_nufft, _, BZ_nufft = cylindrical_components(xyz_s, B_nufft[mask])

        stride = max(1, int(np.ceil(mask.sum() / max_arrows)))
        sample = np.arange(mask.sum())[::stride]
        lines = flux_surface_lines(eq, LinearGrid, float(zeta), rho_values)

        # Outside-LCFS arrows for this section (NUFFT model only).
        o_R = o_Z = o_BR = o_BZ = None
        if has_outside:
            omask = np.isclose(out_zeta, zeta, rtol=0.0, atol=1e-9)
            if omask.any():
                o_xyz = out_xyz[omask]
                o_R = np.sqrt(o_xyz[:, 0] ** 2 + o_xyz[:, 1] ** 2)
                o_Z = o_xyz[:, 2]
                o_BR, _, o_BZ = cylindrical_components(o_xyz, out_B[omask])

        # Shared NUFFT-panel color range (inside + outside), robust to coil
        # field spikes far outside via a 95th-percentile vmax.
        Un, Vn, magn = unit(BR_nufft, BZ_nufft)
        nufft_mags = [magn[sample]]
        if o_R is not None:
            _, _, o_mag = unit(o_BR, o_BZ)
            nufft_mags.append(o_mag)
        all_mag = np.concatenate(nufft_mags)
        vmin = float(all_mag.min())
        vmax = float(np.percentile(all_mag, 95))
        if vmax <= vmin:
            vmax = float(all_mag.max()) or vmin + 1e-12

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)

        # DESC panel (inside only).
        ax = axes[0]
        for _, line_R, line_Z in lines:
            ax.plot(line_R, line_Z, color="0.35", lw=0.9)
        Ud, Vd, magd = unit(BR_desc, BZ_desc)
        ax.quiver(
            R[sample], Z[sample], Ud[sample], Vd[sample], magd[sample],
            angles="xy", scale_units="xy", scale=qscale, width=qwidth, cmap="viridis",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("DESC")
        ax.set_xlabel("R [m]")
        ax.grid(True, alpha=0.18)

        # NUFFT panel (inside + outside the LCFS).
        ax = axes[1]
        for _, line_R, line_Z in lines:
            ax.plot(line_R, line_Z, color="0.35", lw=0.9)
        q = ax.quiver(
            R[sample], Z[sample], Un[sample], Vn[sample], magn[sample],
            angles="xy", scale_units="xy", scale=qscale, width=qwidth, cmap="viridis",
        )
        q.set_clim(vmin, vmax)
        if o_R is not None:
            Uo, Vo, o_mag = unit(o_BR, o_BZ)
            qo = ax.quiver(
                o_R, o_Z, Uo, Vo, o_mag,
                angles="xy", scale_units="xy", scale=qscale, width=qwidth,
                cmap="viridis", alpha=0.9,
            )
            qo.set_clim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(model_label + (" (+ outside LCFS)" if o_R is not None else ""))
        ax.set_xlabel("R [m]")
        ax.grid(True, alpha=0.18)

        axes[0].set_ylabel("Z [m]")
        fig.colorbar(q, ax=axes, shrink=0.82, label="sqrt(BR^2 + BZ^2) [T]")
        fig.suptitle(
            f"{configuration_label} in-plane B quiver, zeta={zeta:.6f} rad, N={N}"
        )
        fig.savefig(outdir / f"quiver_zeta{section_index:02d}_N{N}.png", dpi=220)
        plt.close(fig)


def plot_component_sections(
    eq,
    LinearGrid,
    plot_data: dict[str, np.ndarray],
    outdir: Path,
    num_cross_sections: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    xyz = plot_data["target_xyz"]
    nodes = plot_data["target_nodes"]
    B_desc = plot_data["B_desc"]
    B_nufft = plot_data["B_nufft"]
    model_label = str(plot_data.get("model_label", "NUFFT"))
    configuration_label = str(plot_data.get("configuration_label", "DESC"))
    N = int(plot_data["N"])
    zetas = select_plot_zetas(nodes, num_cross_sections)
    rho_values = (0.25, 0.5, 0.75, 1.0)

    for section_index, zeta in enumerate(zetas):
        mask = np.isclose(nodes[:, 2], zeta, rtol=0.0, atol=1e-12)
        xyz_s = xyz[mask]
        R = np.sqrt(xyz_s[:, 0] ** 2 + xyz_s[:, 1] ** 2)
        Z = xyz_s[:, 2]
        tri = mtri.Triangulation(R, Z)
        desc_BR, desc_Bphi, desc_BZ = cylindrical_components(xyz_s, B_desc[mask])
        nufft_BR, nufft_Bphi, nufft_BZ = cylindrical_components(xyz_s, B_nufft[mask])
        components = [
            ("BR [T]", desc_BR, nufft_BR),
            ("BZ [T]", desc_BZ, nufft_BZ),
            ("Bphi [T]", desc_Bphi, nufft_Bphi),
        ]
        lines = flux_surface_lines(eq, LinearGrid, float(zeta), rho_values)

        # Build a clip path from the LCFS (last entry, rho=1) so that the
        # Delaunay outer triangles from tricontourf don't bleed outside the
        # plasma boundary.
        lcfs_R, lcfs_Z = lines[-1][1], lines[-1][2]
        lcfs_verts = np.column_stack([lcfs_R, lcfs_Z])
        lcfs_codes = (
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(lcfs_verts) - 2)
            + [MplPath.CLOSEPOLY]
        )
        lcfs_mpl_path = MplPath(lcfs_verts, lcfs_codes)

        fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.0), sharex=True, sharey=True)
        for row, (label, desc_vals, nufft_vals) in enumerate(components):
            diff = nufft_vals - desc_vals
            vmin, vmax = np.percentile(np.r_[desc_vals, nufft_vals], [2, 98])
            if vmin == vmax:
                vmin, vmax = float(np.min(desc_vals)), float(np.max(desc_vals))
            err_lim = np.percentile(np.abs(diff), 98)
            err_lim = float(err_lim if err_lim > 0 else np.max(np.abs(diff)))

            panels = [
                ("DESC", desc_vals, "viridis", vmin, vmax),
                (model_label, nufft_vals, "viridis", vmin, vmax),
                (f"{model_label} - DESC", diff, "coolwarm", -err_lim, err_lim),
            ]
            for col, (title, vals, cmap, panel_vmin, panel_vmax) in enumerate(panels):
                ax = axes[row, col]
                cont = ax.tricontourf(
                    tri,
                    vals,
                    levels=40,
                    cmap=cmap,
                    vmin=panel_vmin,
                    vmax=panel_vmax,
                )
                clip = PathPatch(lcfs_mpl_path, transform=ax.transData, visible=False)
                ax.add_patch(clip)
                # matplotlib >= 3.8 removed ContourSet.collections; the
                # ContourSet itself is now the artist to clip.
                if hasattr(cont, "collections"):
                    for col_obj in cont.collections:
                        col_obj.set_clip_path(clip)
                else:
                    cont.set_clip_path(clip)
                for _, line_R, line_Z in lines:
                    ax.plot(line_R, line_Z, color="k", lw=0.65, alpha=0.45)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.12)
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(f"{label}\nZ [m]")
                fig.colorbar(cont, ax=ax, shrink=0.78)
        for ax in axes[-1, :]:
            ax.set_xlabel("R [m]")
        fig.suptitle(
            f"{configuration_label} B components, zeta={zeta:.6f} rad, N={N}"
        )
        fig.tight_layout()
        fig.savefig(outdir / f"components_zeta{section_index:02d}_N{N}.png", dpi=220)
        plt.close(fig)


def plot_lcfs_zoom(
    jax,
    jnp,
    modules,
    eq,
    LinearGrid,
    source,
    box_side: float,
    center: np.ndarray,
    n: int,
    external_field,
    args: argparse.Namespace,
    plot_data: dict[str, np.ndarray],
    outdir: Path,
) -> None:
    """Zoom on the LCFS to look for Gibbs ringing in the NUFFT field.

    For each plotted cross section this evaluates BR, BZ and Bphi on a dense
    band of flux surfaces just inside the boundary (rho in
    [``--lcfs-zoom-rho-min``, 1.0]) and plots them against the poloidal angle,
    DESC on the left and the NUFFT volume(+coils) model on the right. The
    spectral Biot-Savart field rings near the LCFS current discontinuity, so any
    Gibbs oscillation shows up as a wiggle in the right column that the smooth
    DESC equilibrium field (left column) does not have.

    The B evaluation grid here is independent of the convergence-scan target
    grid: it is concentrated near the edge and finely sampled in theta so the
    oscillations are actually resolved.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    nodes = plot_data["target_nodes"]
    zetas = select_plot_zetas(nodes, args.num_cross_sections)
    rho_min = float(args.lcfs_zoom_rho_min)
    rho_vals = np.linspace(rho_min, 1.0, args.lcfs_zoom_n_rho)

    # Dense near-edge evaluation grid at the plotted cross sections.
    edge_grid = LinearGrid(
        rho=rho_vals,
        theta=args.lcfs_zoom_n_theta,
        zeta=np.asarray(zetas, dtype=float),
        NFP=eq.NFP,
        sym=False,
        axis=False,
    )
    edge_data = eq.compute(["X", "Y", "Z", "B"], grid=edge_grid, basis="xyz")
    edge_xyz = np.column_stack(
        [
            np.asarray(edge_data["X"]),
            np.asarray(edge_data["Y"]),
            np.asarray(edge_data["Z"]),
        ]
    )
    B_desc = np.asarray(edge_data["B"])
    edge_nodes = np.asarray(edge_grid.nodes)

    # NUFFT plasma field on the same targets. B_hat is not retained from the
    # scan loop, so recompute it once at the plotted box/source resolution.
    X, Y, Z, Jx, Jy, Jz, w = source
    cj = jnp.asarray(center)
    box = modules["BoxParams"](box_side, box_side, box_side, n, n, n)
    Bx_hat, By_hat, Bz_hat = modules["compute_B_hat"](
        X - cj[0], Y - cj[1], Z - cj[2], Jx, Jy, Jz, w, box, eps=args.eps,
        spectral_filter=args.spectral_filter, filter_order=args.filter_order,
    )
    pos = jnp.asarray(edge_xyz) - cj
    Bx, By, Bz = modules["eval_B"](Bx_hat, By_hat, Bz_hat, pos, box, eps=args.eps)
    B_nufft = np.asarray(jnp.stack([Bx, By, Bz], axis=1))
    if external_field is not None:
        B_ext = np.asarray(
            external_field.compute_magnetic_field(
                jnp.asarray(edge_xyz), basis="xyz", chunk_size=args.coil_chunk_size
            )
        )
        B_nufft = B_nufft + B_ext

    model_label = str(plot_data.get("model_label", "NUFFT"))
    configuration_label = str(plot_data.get("configuration_label", "DESC"))

    desc_BR, desc_Bphi, desc_BZ = cylindrical_components(edge_xyz, B_desc)
    nufft_BR, nufft_Bphi, nufft_BZ = cylindrical_components(edge_xyz, B_nufft)
    components = [
        ("$B_R$ [T]", desc_BR, nufft_BR),
        ("$B_Z$ [T]", desc_BZ, nufft_BZ),
        (r"$B_\phi$ [T]", desc_Bphi, nufft_Bphi),
    ]

    norm = Normalize(vmin=rho_min, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    def surface_curve(mask, vals):
        """theta-sorted, loop-closed (theta, value) for one flux surface."""
        theta = edge_nodes[mask, 1]
        order = np.argsort(theta)
        th = theta[order]
        v = vals[mask][order]
        return np.r_[th, th[0] + 2.0 * np.pi], np.r_[v, v[0]]

    for section_index, zeta in enumerate(zetas):
        zmask = np.isclose(edge_nodes[:, 2], zeta, rtol=0.0, atol=1e-9)
        fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.5), sharex=True)
        for row, (label, desc_vals, nufft_vals) in enumerate(components):
            row_lo, row_hi = np.inf, -np.inf
            for col, (title, vals) in enumerate(
                [("DESC", desc_vals), (model_label, nufft_vals)]
            ):
                ax = axes[row, col]
                for rho in rho_vals:
                    rmask = zmask & np.isclose(
                        edge_nodes[:, 0], rho, rtol=0.0, atol=1e-9
                    )
                    if not rmask.any():
                        continue
                    th, v = surface_curve(rmask, vals)
                    ax.plot(th, v, color=cmap(norm(rho)), lw=1.0)
                    row_lo = min(row_lo, float(v.min()))
                    row_hi = max(row_hi, float(v.max()))
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(label)
                ax.grid(True, alpha=0.2)
            pad = 0.05 * (row_hi - row_lo if row_hi > row_lo else 1.0)
            for col in range(2):
                axes[row, col].set_ylim(row_lo - pad, row_hi + pad)
        for ax in axes[-1, :]:
            ax.set_xlabel(r"poloidal angle $\theta$ [rad]")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(
            sm, ax=axes.ravel().tolist(), shrink=0.82,
            label=r"flux surface $\rho$ (outer = closer to LCFS)",
        )
        fig.suptitle(
            f"{configuration_label} near-LCFS B components, "
            f"zeta={zeta:.6f} rad, N={n}\n"
            rf"$\rho \in [{rho_min:g}, 1.0]$ — Gibbs ringing shows as "
            r"$\theta$ oscillation in the NUFFT column near $\rho=1$"
        )
        fig.savefig(outdir / f"lcfs_zoom_zeta{section_index:02d}_N{n}.png", dpi=220)
        plt.close(fig)

    # Quantify the edge ringing on the LCFS (rho=1) for each cross section.
    lcfs_mask_all = np.isclose(edge_nodes[:, 0], 1.0, rtol=0.0, atol=1e-9)
    for section_index, zeta in enumerate(zetas):
        mask = lcfs_mask_all & np.isclose(edge_nodes[:, 2], zeta, rtol=0.0, atol=1e-9)
        if not mask.any():
            continue
        resid = np.linalg.norm(B_nufft[mask] - B_desc[mask], axis=1)
        ref = np.sqrt(np.mean(np.sum(B_desc[mask] ** 2, axis=1)))
        rel = float(resid.max() / ref) if ref > 0 else float("nan")
        print(
            f"  LCFS zoom zeta={zeta:.4f}: max |B_NUFFT - B_DESC| on rho=1 "
            f"= {resid.max():.4e} T ({rel:.3%} of |B|)"
        )

    del box, Bx_hat, By_hat, Bz_hat, Bx, By, Bz
    gc.collect()


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    repo_root: Path,
    desc,
    jax,
    box_side: float,
    center: np.ndarray,
    num_source: int,
    num_target: int,
) -> None:
    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "repo_root": str(repo_root),
        "desc_version": getattr(desc, "__version__", "unknown"),
        "desc_file": getattr(desc, "__file__", "unknown"),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "box_side": float(box_side),
        "box_center_xyz": center.tolist(),
        "num_source_points_after_nfp_replication": int(num_source),
        "num_target_points": int(num_target),
        "note": (
            "This compares DESC total B with the NUFFT Biot-Savart field from "
            "the selected plasma source model plus an optional direct DESC "
            "coil/external field. For free-boundary cases, provide --coil-file "
            "so the model field is B_NUFFT_plasma + B_coils."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def run_source_scan(
    args,
    modules,
    jax,
    jnp,
    eq,
    repo_root,
    desc,
    target_grid,
    target_xyz: np.ndarray,
    B_desc: np.ndarray,
    B_external,
    external_field,
    model_label: str,
) -> None:
    """Diagnostic 2: scan source resolution at fixed box N."""
    grids = parse_source_grids(args.source_grids)
    n = args.source_scan_n
    scan_csv = args.outdir / "source_scan_metrics.csv"
    if scan_csv.exists():
        scan_csv.unlink()

    print(f"Source-resolution scan at fixed box N={n} over grids: {grids}")

    rows: list[dict[str, float | int | str]] = []
    plot_data = None
    last_source = None
    last_box_side = float(args.padding)
    last_center = np.zeros(3)
    last_n_source = 0

    for (n_rho, n_theta, n_zeta) in grids:
        row: dict[str, float | int | str] = {
            "n_rho": n_rho,
            "n_theta": n_theta,
            "n_zeta": n_zeta,
            "source_points": n_rho * n_theta * n_zeta,
            "status": "ok",
        }
        try:
            print(f"Source grid (rho,theta,zeta)=({n_rho},{n_theta},{n_zeta})...")
            source = extract_source(args, modules, eq, n_rho, n_theta, n_zeta)
            source_xyz = np.column_stack([np.asarray(s) for s in source[:3]])
            box_side, center = box_geometry(
                source_xyz[:, 0], source_xyz[:, 1], source_xyz[:, 2], args.padding
            )
            B_model, B_plasma, t_hat, t_eval = run_nufft_once(
                jax, jnp, modules, source, box_side, center, n,
                target_xyz, B_external, args.eps,
                spectral_filter=args.spectral_filter, filter_order=args.filter_order,
            )
            row["compute_B_hat_s"] = t_hat
            row["eval_B_s"] = t_eval
            row.update(field_metrics(B_desc, B_model))
            append_source_scan_csv(scan_csv, row)
            rows.append(row)
            print(
                f"  source_points={row['source_points']}: "
                f"rel_rms={row['rel_rms']:.6e}, |B| rel_rms={row['mag_rel_rms']:.6e}"
            )
            last_source = source
            last_box_side, last_center = box_side, center
            last_n_source = source_xyz.shape[0]
            plot_data = {
                "N": np.array(n),
                "target_nodes": np.asarray(target_grid.nodes),
                "target_xyz": target_xyz,
                "B_desc": B_desc,
                "B_nufft": B_model,
                "B_plasma_nufft": B_plasma,
                "B_external": (
                    np.zeros_like(B_model) if B_external is None else B_external
                ),
                "configuration_label": args.config_name,
                "model_label": model_label,
            }
        except Exception as err:
            row["status"] = "failed"
            row["error"] = repr(err)
            append_source_scan_csv(scan_csv, row)
            rows.append(row)
            print(f"  source grid failed: {err!r}")
            if not args.keep_going:
                raise

    plot_source_scan(rows, args.outdir, args.config_name, n)

    write_run_metadata(
        args.outdir / "run_metadata.json",
        args,
        repo_root,
        desc,
        jax,
        last_box_side,
        last_center,
        last_n_source,
        target_xyz.shape[0],
    )

    # Field-quality plots use the finest successful source grid.
    if plot_data is not None:
        try:
            compute_outside_quiver(
                plot_data, jax, jnp, modules, eq, modules["LinearGrid"],
                last_source, last_box_side, last_center, n, external_field, args,
            )
        except Exception as err:
            print(f"Skipping outside-LCFS quiver field: {err!r}")
        plot_quiver_sections(
            eq,
            modules["LinearGrid"],
            plot_data,
            args.outdir,
            args.max_quiver_arrows,
            args.num_cross_sections,
        )
        plot_component_sections(
            eq,
            modules["LinearGrid"],
            plot_data,
            args.outdir,
            args.num_cross_sections,
        )
        if args.lcfs_zoom:
            try:
                plot_lcfs_zoom(
                    jax, jnp, modules, eq, modules["LinearGrid"],
                    last_source, last_box_side, last_center, n,
                    external_field, args, plot_data, args.outdir,
                )
            except Exception as err:
                print(f"Skipping LCFS zoom plot: {err!r}")


def main() -> None:
    args = parse_args()
    repo_root = configure_paths(args)
    jax, jnp = configure_jax(args)
    modules = load_runtime_modules()

    desc = modules["desc"]
    desc_load = modules["desc_load"]
    LinearGrid = modules["LinearGrid"]

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark configuration: {args.config_name}")
    print(f"Equilibrium: {args.eq_file}")
    print(f"Coils:       {args.coil_file or '(none)'}")
    print("Loading DESC equilibrium...")
    eq = load_equilibrium(args, desc_load)
    external_field = load_external_field(args, desc_load)
    validate_external_field(eq, external_field, args)
    print(
        f"Loaded equilibrium: L={eq.L}, M={eq.M}, N={eq.N}, "
        f"NFP={eq.NFP}, sym={eq.sym}"
    )
    if external_field is not None:
        print(f"Loaded coil field from {args.coil_file}")

    print("Computing DESC target B on the evaluation grid...")
    target_grid = LinearGrid(
        rho=target_rho(args.n_rho, args.target_rho_max),
        theta=args.n_theta,
        zeta=args.n_zeta,
        NFP=eq.NFP,
        sym=False,
        axis=False,
    )
    target_data = eq.compute(["X", "Y", "Z", "B"], grid=target_grid, basis="xyz")
    target_xyz = np.column_stack(
        [
            np.asarray(target_data["X"]),
            np.asarray(target_data["Y"]),
            np.asarray(target_data["Z"]),
        ]
    )
    B_desc = np.asarray(target_data["B"])

    B_external = None
    if external_field is not None:
        print("Computing external/coils B on the target grid...")
        B_external_jax = external_field.compute_magnetic_field(
            jnp.asarray(target_xyz),
            basis="xyz",
            chunk_size=args.coil_chunk_size,
        )
        block_until_ready(jax, B_external_jax)
        B_external = np.asarray(B_external_jax)
        B_external_norm = np.sqrt(np.mean(np.sum(B_external**2, axis=1)))
        print(f"External/coils B RMS on target grid: {B_external_norm:.6e} T")

    source_label = (
        "NUFFT volume" if args.source_model == "volume" else "NUFFT boundary"
    )
    model_label = source_label if B_external is None else f"{source_label} + coils"

    if args.source_scan:
        run_source_scan(
            args, modules, jax, jnp, eq, repo_root, desc,
            target_grid, target_xyz, B_desc, B_external, external_field, model_label,
        )
        print(f"Done. Outputs written to {args.outdir}")
        return

    # --- box-N convergence scan ---
    # In a joint scan the source resolution co-varies with N (the matched
    # refinement path); otherwise the source is fixed and shared across N.
    if args.joint_scan:
        scan_items = parse_joint_grids(args.joint_grids)
        print(f"Joint scan over (N, n_rho, n_theta, n_zeta): {scan_items}")
    else:
        scan_items = [
            (n, args.n_rho, args.n_theta, args.n_zeta) for n in args.n_values
        ]

    metrics_path = args.outdir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    rows: list[dict[str, float | int | str]] = []
    plot_data = None
    requested_plot_n_seen = False
    source = None
    last_dims = None
    box_side = None
    center = None
    metadata_written = False

    for (n, n_rho, n_theta, n_zeta) in scan_items:
        row: dict[str, float | int | str] = {"N": int(n), "status": "ok"}
        try:
            if (n_rho, n_theta, n_zeta) != last_dims:
                print(f"Extracting DESC source current ({n_rho},{n_theta},{n_zeta})...")
                source = extract_source(args, modules, eq, n_rho, n_theta, n_zeta)
                source_xyz = np.column_stack([np.asarray(s) for s in source[:3]])
                box_side, center = box_geometry(
                    source_xyz[:, 0], source_xyz[:, 1], source_xyz[:, 2], args.padding
                )
                last_dims = (n_rho, n_theta, n_zeta)
                if not metadata_written:
                    write_run_metadata(
                        args.outdir / "run_metadata.json",
                        args, repo_root, desc, jax, box_side, center,
                        source_xyz.shape[0], target_xyz.shape[0],
                    )
                    metadata_written = True
            print(f"Running box N={n} (source {n_rho},{n_theta},{n_zeta})...")
            B_model, B_plasma_nufft, t_hat, t_eval = run_nufft_once(
                jax, jnp, modules, source, box_side, center, n,
                target_xyz, B_external, args.eps,
                spectral_filter=args.spectral_filter, filter_order=args.filter_order,
            )
            row["compute_B_hat_s"] = t_hat
            row["eval_B_s"] = t_eval
            row.update(field_metrics(B_desc, B_model))
            append_metrics_csv(metrics_path, row)
            rows.append(row)

            print(
                f"N={n}: rel_l2={row['rel_l2']:.6e}, "
                f"rel_rms={row['rel_rms']:.6e}, "
                f"|B| rel_rms={row['mag_rel_rms']:.6e}"
            )

            if args.plot_n is None or n == args.plot_n:
                requested_plot_n_seen = requested_plot_n_seen or n == args.plot_n
                plot_data = {
                    "N": np.array(n),
                    "target_nodes": np.asarray(target_grid.nodes),
                    "target_xyz": target_xyz,
                    "B_desc": B_desc,
                    "B_nufft": B_model,
                    "B_plasma_nufft": B_plasma_nufft,
                    "B_external": (
                        np.zeros_like(B_model) if B_external is None else B_external
                    ),
                    "configuration_label": args.config_name,
                    "model_label": model_label,
                }
        except Exception as err:
            row["status"] = "failed"
            row["error"] = repr(err)
            append_metrics_csv(metrics_path, row)
            rows.append(row)
            print(f"N={n} failed: {err!r}")
            if not args.keep_going:
                raise

    plot_error_scan(rows, args.outdir, args.config_name, x_cubed=args.joint_scan)

    if plot_data is not None and (args.plot_n is None or requested_plot_n_seen):
        np.savez_compressed(
            args.outdir / f"desc_bfield_sample_N{int(plot_data['N'])}.npz",
            target_nodes=plot_data["target_nodes"],
            target_xyz=plot_data["target_xyz"],
            B_desc=plot_data["B_desc"],
            B_nufft=plot_data["B_nufft"],
            B_plasma_nufft=plot_data["B_plasma_nufft"],
            B_external=plot_data["B_external"],
            configuration_label=np.array(plot_data["configuration_label"]),
            center=center,
        )
        try:
            compute_outside_quiver(
                plot_data, jax, jnp, modules, eq, LinearGrid,
                source, box_side, center, int(plot_data["N"]),
                external_field, args,
            )
        except Exception as err:
            print(f"Skipping outside-LCFS quiver field: {err!r}")
        plot_quiver_sections(
            eq,
            LinearGrid,
            plot_data,
            args.outdir,
            args.max_quiver_arrows,
            args.num_cross_sections,
        )
        plot_component_sections(
            eq,
            LinearGrid,
            plot_data,
            args.outdir,
            args.num_cross_sections,
        )
        if args.lcfs_zoom:
            try:
                plot_lcfs_zoom(
                    jax, jnp, modules, eq, LinearGrid,
                    source, box_side, center, int(plot_data["N"]),
                    external_field, args, plot_data, args.outdir,
                )
            except Exception as err:
                print(f"Skipping LCFS zoom plot: {err!r}")
    elif args.plot_n is not None:
        print(f"No plot data saved because requested --plot-n {args.plot_n} did not finish.")

    print(f"Done. Outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
