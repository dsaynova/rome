#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2023-22-1120
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=causal-tracing-gpt2-xl
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/pararel_gpt2_xl_%A.out
#SBATCH -t 0-04:00:00

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

DATA_DIR="/cephyr/users/lovhag/Alvis/projects/rome/data"

python -m experiments.causal_trace_pararel \
    --model_name "gpt2-xl" \
    --fact_file "${DATA_DIR}/P19_gpt2_xl_preds.jsonl" \
    --output_dir "${DATA_DIR}/results/gpt2-xl/P19/causal_trace_pararel_${SLURM_JOB_ID}"