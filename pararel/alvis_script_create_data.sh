#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A NAISS2023-22-1120
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=causal-tracing-data-gen-gpt2-xl
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/create_data_pararel_gpt2_xl_%A.out
#SBATCH -t 0-04:00:00

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

RELATION="P103"

python -m pararel.create_pararel_dsets --model_name gpt2-xl --relation $RELATION --output_folder data --pararel_data_path "/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space"