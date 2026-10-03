#!/bin/bash
#SBATCH --job-name=sdw_reflect_L16
#SBATCH --time=30:00:00
#SBATCH --output=reflective_sde_%j.out
#SBATCH --error=reflective_sde_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=cherngroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G

set -euo pipefail

module purge
module load julia

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIRECTORY}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SDW_TELEC="${SDW_TELEC:-0.01}"
export SDW_TFLUC="${SDW_TFLUC:-${SDW_TELEC}}"
export SDW_NRUNS="${SDW_NRUNS:-10000}"
export SDW_OUTPUT_DIR="${SDW_OUTPUT_DIR:-${SCRIPT_DIRECTORY}/outputs}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date -Is)"
echo "T_elec=${SDW_TELEC}, T_fluc=${SDW_TFLUC}, steps=${SDW_NRUNS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

stdbuf -oL -eL julia --startup-file=no --color=yes \
    Neel16_direct_M_reflected.jl

echo "Finished: $(date -Is)"
