#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2023-22-1120
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=causal-tracing-fact-recall-gpt2-xl
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/fact_recall_gpt2_xl_%A.out
#SBATCH -t 0-24:00:00

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

DATA_DIR="/cephyr/users/lovhag/Alvis/projects/rome/data"

python -m pararel.fact_recall_detection.causal_trace \
    --model_name "gpt2-xl" \
    --fact_file "/cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_preds.jsonl" \
    --output_dir "/cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_causal_trace_${SLURM_JOB_ID}" \
    --only_te \