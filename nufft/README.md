## Installation (with [uv](https://github.com/astral-sh/uv))

`uv` is a fast Python package/environment manager developed by Astral.

### 1. Install `uv`
If you don’t already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or run `source ~/.bashrc` / `source ~/.zshrc`).

Check installation:
```bash
uv --version
```

---

### 2. Clone the repository

```bash
git clone https://github.com/your-username/nufft_biot.git
cd nufft_biot
```

---

### 3. Create and sync environment

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml`.

To activate the environment:
```bash
uv run python
```
or for any command:
```bash
uv run <command>
```

## Run tests

```bash
uv run pytest -v
```

## DESC stellarator B-field benchmark

The benchmark driver is in `benchmarks/benchmark_desc_bfield.py`. It is
configuration-driven: each stellarator case is a JSON file in
`benchmarks/configurations/`, selected with `--config <name>`. The default
`precise_QA` configuration (validated at `N=64`) scans
`Nx=Ny=Nz = 64, 128, 256, 512` with `padding=2.0`, `eps=1e-12`, and a
`16 x 32 x 64` DESC source/target grid; a `W7-X` configuration is also provided.
The driver compares DESC total `B` with `B_NUFFT(plasma J) + B_coils` and writes
error metrics plus the requested quiver/component plots.

See `benchmarks/README.md`, `benchmarks/submit_desc_bfield_scan.py`,
`benchmarks/aggregate_error_scan.py`, and `benchmarks/slurm_desc_bfield.sbatch`
for cluster usage.

