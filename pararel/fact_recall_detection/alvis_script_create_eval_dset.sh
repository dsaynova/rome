#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2023-22-1120
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=causal-tracing-data-gen-gpt2-xl
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/create_data_fact_recall_gpt2_xl_%A.out
#SBATCH -t 0-04:00:00

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m pararel.fact_recall_detection.create_eval_dset \
    --model_name gpt2-xl \
    --output_folder data/fact_recall_detection \
    --data_path "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/data_creation/final_splits/confident_fact_recall_preds.jsonl" \