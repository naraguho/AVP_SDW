#!/bin/bash
#SBATCH --job-name=sdw_xsib_test
#SBATCH --time=00:30:00
#SBATCH --output=short_xsib_%j.out
#SBATCH --error=short_xsib_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --account=cherngroup
#SBATCH --gres=gpu:a6000:1
#SBATCH --constraint=a6000
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sgv2ew@virginia.edu

set -euo pipefail

module purge
module load julia
module load cuda/10.2.89

cd "${SLURM_SUBMIT_DIR}"

# The Julia source reads these two variables. Twenty steps produce snapshots
# at steps 10 and 20 and are sufficient for the first integration smoke test.
export SDW_NRUNS=20
export SDW_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/short_xheun_sib_${SLURM_JOB_ID}"

mkdir -p "${SDW_OUTPUT_DIR}"

echo "Host: $(hostname)"
echo "Julia: $(julia --version)"
echo "Output: ${SDW_OUTPUT_DIR}"
nvidia-smi

julia sdw_evolution_cuda.jl

echo "Short validation completed successfully."
