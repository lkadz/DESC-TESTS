#!/bin/bash
#SBATCH --job-name=nufft-desc-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nufft-desc-b-%j.out
#SBATCH --error=logs/nufft-desc-b-%j.err

set -eo pipefail
export PS1="${PS1-}"

# Edit these for the Princeton cluster environment you use.
# module purge
# module load anaconda3/2024.10
# module load cuda/12.4
# conda activate desc-env
# set -u
# or source /path/to/venv/bin/activate, then set -u after activation

# Pick the stellarator configuration to run (see benchmarks/configurations/).
CONFIG=precise_QA

mkdir -p logs "results/desc_bfield/${CONFIG}"

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The configuration file supplies the equilibrium, coils, source model, and
# grid defaults. Set CONFIG=W7-X (with a paired coil file) to run another case.
# Running all N values in one process writes a multi-point error_vs_N.png; the
# separate-job scan instead uses aggregate_error_scan.py.
python benchmarks/benchmark_desc_bfield.py \
  --config "${CONFIG}" \
  --desc-root auto \
  --outdir "results/desc_bfield/${CONFIG}" \
  --n-values 64 128 256 512 \
  --jax-platform cuda
