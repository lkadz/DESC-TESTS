#!/bin/bash
#SBATCH --job-name=desc-mockup-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --output=logs/02_coil_optimization-%j.out
#SBATCH --error=logs/02_coil_optimization-%j.err

set -eo pipefail

module purge
module load anaconda3/2024.10
export PS1="${PS1-}"
conda activate desc-env

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

cd "$(dirname "$0")"
mkdir -p logs

python 02_coil_optimization.py
