#!/bin/bash

# make sure that you have done the following before running this script:
# module purge
# module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
# source venv/bin/activate

python -m pararel.fact_recall_detection.process_data \
    --output_file /cephyr/users/lovhag/Alvis/projects/rome/data/fact_recall_detection/gpt2_xl_final.csv \
    --queries_data_file /cephyr/users/lovhag/Alvis/projects/rome/data/fact_recall_detection/gpt2_xl_preds.jsonl \
    --ct_data_folder /cephyr/users/lovhag/Alvis/projects/rome/data/fact_recall_detection/gpt2_xl_causal_trace \