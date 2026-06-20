#!/bin/bash
# Wrapper that ensures benchmark_desc_bfield.py runs inside desc-env.
# Usage: bash benchmarks/run_benchmark.sh [any benchmark_desc_bfield.py args]

set -eo pipefail

CONDA_ENV="desc-env"

# Source conda so that 'conda activate' works in non-interactive shells.
CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "ERROR: conda not found. Load the anaconda module first:" >&2
    echo "  module load anaconda3/2024.10" >&2
    exit 1
}
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

exec python benchmarks/benchmark_desc_bfield.py "$@"
