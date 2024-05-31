#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2024-22-264
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=causal-tracing-gpt2-xl-guesswork-set
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/causal_tracing_gpt2_xl_guesswork_set_%A.out
#SBATCH -t 2-00:00:00

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m pararel.eval_on_fact_recall_set.causal_trace \
    --model_name "gpt2-xl" \
    --fact_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_guesswork.json" \
    --output_dir "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_guesswork_${SLURM_JOB_ID}" 