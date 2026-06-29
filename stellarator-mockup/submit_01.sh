#!/bin/bash
#SBATCH --job-name=desc-mockup-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --output=logs/01_fixed_boundary.out
#SBATCH --error=logs/01_fixed_boundary.err

set -eo pipefail

module purge
module load anaconda3/2024.10
export PS1="${PS1-}"
conda activate desc-env

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

python 01_fixed_boundary.py

jobstats $SLURM_JOB_ID
