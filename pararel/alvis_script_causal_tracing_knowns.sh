#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2023-22-1120
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=causal-tracing-gpt2-xl-knowns
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/causal_tracing_gpt2_xl_knowns_%A.out
#SBATCH -t 4-00:00:00

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

DATA_DIR="/cephyr/users/lovhag/Alvis/projects/rome/data"

python -m experiments.causal_trace \
    --model_name "gpt2-xl" \
    --fact_file "${DATA_DIR}/known_1000.json" \
    --output_dir "${DATA_DIR}/results/gpt2-xl/known_1000/causal_trace_${SLURM_JOB_ID}" 