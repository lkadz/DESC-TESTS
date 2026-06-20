"""Compare the plasma current density before and after the edge taper.

The edge taper (``benchmarks/.../edge_taper``) multiplies the DESC volume
current J by a smooth window that rolls to zero between ``--edge-taper-rho0``
and the LCFS, removing the current jump that causes Gibbs ringing in the
spectral Biot-Savart field. This script visualizes what that does to J:

- a radial profile of shell-mean |J| before vs after, with the taper window
- an R-Z cross-section of |J| before / after / removed
- the fraction of volume-integrated |J| removed (the source of the field bias)

Only DESC current extraction is used (no NUFFT), so this runs fine on a login
node with ``--jax-platform cpu``.

Example:

    python benchmarks/compare_current_taper.py \
      --config 2bump_n0_0.07_n1_0.02_k_iota_-1.0 \
      --edge-taper-rho0 0.95 --jax-platform cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import benchmark_desc_bfield as bench
from bench_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DESC plasma current density before/after the edge taper."
    )
    parser.add_argument("--config", default="2bump_n0_0.07_n1_0.02_k_iota_-1.0")
    parser.add_argument("--desc-root", default="auto")
    parser.add_argument(
        "--eq-file", type=Path, default=None, help="Override the config equilibrium."
    )
    parser.add_argument("--n-rho", type=int, default=32)
    parser.add_argument("--n-theta", type=int, default=64)
    parser.add_argument("--n-zeta", type=int, default=128)
    parser.add_argument("--edge-taper-rho0", type=float, default=0.95)
    parser.add_argument(
        "--edge-taper-shape",
        choices=("smoothstep", "smootherstep", "cosine", "quadratic"),
        default="smoothstep",
    )
    parser.add_argument("--num-cross-sections", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--x64", default=True, action=argparse.BooleanOptionalAction,
        help="Enable JAX x64.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=("auto", "cpu", "gpu", "cuda", "rocm", "tpu"),
        default="auto",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.eq_file is None:
        args.eq_file = config["eq_file"]
    args.config_name = config["name"]
    return args


def plot_profile(rhos, prof_before, prof_after, window, removed, args, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(rhos, prof_before, "o-", color="C0", label="|J| before")
    ax.plot(rhos, prof_after, "s-", color="C1", label="|J| after taper")
    ax.axvline(args.edge_taper_rho0, color="0.6", ls="--", lw=1.0)
    ax.set_xlabel("rho")
    ax.set_ylabel("shell-mean |J|  [A/m^2]")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(rhos, window, "k:", lw=1.6, alpha=0.7, label="taper window")
    ax2.set_ylabel("taper window")
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    ax.set_title(
        f"{args.config_name}: current density taper "
        f"(rho0={args.edge_taper_rho0}, {args.edge_taper_shape})\n"
        f"{removed * 100:.2f}% of volume-integrated |J| removed"
    )
    fig.tight_layout()
    path = outdir / "current_profile_taper.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_cross_sections(eq, LinearGrid, nodes, R, Z, Jmag, Jmag_after, args, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    zetas = bench.select_plot_zetas(nodes, args.num_cross_sections)
    paths = []
    for section_index, zeta in enumerate(zetas):
        mask = np.isclose(nodes[:, 2], zeta, rtol=0.0, atol=1e-12)
        Rs, Zs = R[mask], Z[mask]
        before, after = Jmag[mask], Jmag_after[mask]
        removed = before - after
        tri = mtri.Triangulation(Rs, Zs)

        lcfs = bench.flux_surface_lines(eq, LinearGrid, float(zeta), (1.0,))[0]
        lcfs_verts = np.column_stack([lcfs[1], lcfs[2]])
        lcfs_codes = (
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(lcfs_verts) - 2)
            + [MplPath.CLOSEPOLY]
        )
        lcfs_path = MplPath(lcfs_verts, lcfs_codes)

        vmax = float(np.percentile(before, 99))
        panels = [
            ("|J| before", before, "viridis", 0.0, vmax),
            ("|J| after taper", after, "viridis", 0.0, vmax),
            ("removed (before - after)", removed, "magma", 0.0, float(np.percentile(removed, 99) or vmax)),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
        for ax, (title, vals, cmap, vmin, vhi) in zip(axes, panels):
            cont = ax.tricontourf(tri, vals, levels=40, cmap=cmap, vmin=vmin, vmax=vhi)
            clip = PathPatch(lcfs_path, transform=ax.transData, visible=False)
            ax.add_patch(clip)
            if hasattr(cont, "collections"):
                for c in cont.collections:
                    c.set_clip_path(clip)
            else:
                cont.set_clip_path(clip)
            ax.plot(lcfs[1], lcfs[2], color="k", lw=0.8, alpha=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)
            ax.set_xlabel("R [m]")
            fig.colorbar(cont, ax=ax, shrink=0.8)
        axes[0].set_ylabel("Z [m]")
        fig.suptitle(
            f"{args.config_name}: |J| cross-section, zeta={zeta:.6f} rad"
        )
        fig.tight_layout()
        path = outdir / f"current_xsection_zeta{section_index:02d}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    bench.configure_paths(args)
    jax, jnp = bench.configure_jax(args)
    modules = bench.load_runtime_modules()
    LinearGrid = modules["LinearGrid"]
    desc_load = modules["desc_load"]
    from nufft_biot.desc_interface import edge_taper

    print(f"Configuration: {args.config_name}")
    print(f"Equilibrium:   {args.eq_file}")
    eq = bench.load_equilibrium(args, desc_load)

    grid = LinearGrid(
        rho=(np.arange(args.n_rho, dtype=float) + 0.5) / args.n_rho,
        theta=args.n_theta,
        zeta=args.n_zeta,
        sym=False,
        NFP=eq.NFP,
        axis=False,
    )
    data = eq.compute(["J", "X", "Y", "Z", "sqrt(g)"], grid=grid, basis="xyz")
    J = np.asarray(data["J"])
    Jmag = np.linalg.norm(J, axis=1)
    rho = np.asarray(grid.nodes)[:, 0]
    dV = np.abs(np.asarray(data["sqrt(g)"]) * np.asarray(grid.weights))

    window = edge_taper(rho, args.edge_taper_rho0, args.edge_taper_shape)
    Jmag_after = Jmag * window  # taper scales all components equally

    int_before = float(np.sum(Jmag * dV))
    int_after = float(np.sum(Jmag_after * dV))
    removed_frac = 1.0 - int_after / int_before if int_before > 0 else float("nan")
    print(
        f"Volume-integrated |J|: before={int_before:.6e}, after={int_after:.6e}, "
        f"removed={removed_frac * 100:.3f}%"
    )

    outdir = args.outdir or (
        Path("results/desc_bfield") / args.config_name / "current_taper"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rhos = np.unique(rho)
    prof_before = np.array([Jmag[rho == r].mean() for r in rhos])
    prof_after = np.array([Jmag_after[rho == r].mean() for r in rhos])
    window_profile = edge_taper(rhos, args.edge_taper_rho0, args.edge_taper_shape)

    import csv

    with (outdir / "current_profile_taper.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rho", "Jmag_before", "Jmag_after", "taper_window"])
        for r, b, a, win in zip(rhos, prof_before, prof_after, window_profile):
            writer.writerow([r, b, a, win])

    profile_path = plot_profile(
        rhos, prof_before, prof_after, window_profile, removed_frac, args, outdir
    )
    print(f"Wrote {profile_path}")

    R = np.sqrt(np.asarray(data["X"]) ** 2 + np.asarray(data["Y"]) ** 2)
    Z = np.asarray(data["Z"])
    xsec_paths = plot_cross_sections(
        eq, LinearGrid, np.asarray(grid.nodes), R, Z, Jmag, Jmag_after, args, outdir
    )
    for path in xsec_paths:
        print(f"Wrote {path}")

    print(f"Done. Outputs in {outdir}")


if __name__ == "__main__":
    main()
