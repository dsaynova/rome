#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2024-22-264
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=causal-tracing-llama2-7B-acc-rank-0-set
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/causal_tracing_gpt2_xl_acc_rank_0_set_llama2_7B_%A.out
#SBATCH -t 2-00:00:00

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source /mimer/NOBACKUP/groups/snic2021-23-309/envs/rome_llama/bin/activate

python -m notebooks.experiments.causal_trace_pararel \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --fact_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/llama2_7B/1000_accurate_rank_0.jsonl" \
    --output_dir "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/causal_trace_acc_rank_0_${SLURM_JOB_ID}" \
    --cache_folder "${TMPDIR}/.cache" \