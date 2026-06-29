#!/bin/bash
#SBATCH --job-name=desc-mockup-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:59:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --output=logs/03_free_boundary.out
#SBATCH --error=logs/03_free_boundary.err

set -eo pipefail

module purge
module load anaconda3/2024.10
export PS1="${PS1-}"
conda activate desc-env

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

python 03_free_boundary.py

sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,ReqMem,State
