#!/bin/bash

# make sure that you have done the following before running this script:
# module purge
# module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
# source venv/bin/activate

RELATION=$1
MODEL="llama7B"

echo "Processing relation ${RELATION} for model ${MODEL}..."


DATA_FOLDER="/cephyr/users/lovhag/Alvis/projects/rome/data"
OUTPUT_FILE="${DATA_FOLDER}/RQ1/${MODEL}/${RELATION}.csv"

QUERIES_FILE="${DATA_FOLDER}/results/${MODEL}/${RELATION}_Llama_2_7b_hf_preds.jsonl"

declare -A ct_data_paths
ct_data_paths["P19"]="${DATA_FOLDER}/results/llama7B/causal_trace_pararel_1915793_P19"
ct_data_paths["P20"]="${DATA_FOLDER}/results/llama7B/causal_trace_pararel_1917992_P20"
ct_data_paths["P27"]="${DATA_FOLDER}/results/llama7B/causal_trace_pararel_1917993_P27"
ct_data_paths["P101"]="${DATA_FOLDER}/results/llama7B/causal_trace_pararel_1917980_P101"
#ct_data_paths["P495"]="${DATA_FOLDER}/results/gpt2-xl/P495/causal_trace_pararel_1883397_P495" -- TO BE ADDED!!
#ct_data_paths["P740"]="${DATA_FOLDER}/results/gpt2-xl/P740/causal_trace_pararel_1883935_P740"
ct_data_paths["P1376"]="${DATA_FOLDER}/results/llama7B/causal_trace_pararel_1917979_P1376"

python -m RQ1.process_data --output_file "${OUTPUT_FILE}" --queries_data_file ${QUERIES_FILE} --ct_data_folder "${ct_data_paths["${RELATION}"]}"