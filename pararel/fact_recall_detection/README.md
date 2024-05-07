# Fact recall detection

Test the TE based approach on our created fact recall detection datasets.

Run our model on the queries and store corresponding TEs using the same approach as rome. 

## 1. Run `alvis_script_create_eval_dset.sh` to get model predictions for each query. (note that the code here should be different from our original - we do not wish to make pairwise comparisons here)

Debug

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

CUDA_VISIBLE_DEVICES=1, python -m pararel.fact_recall_detection.create_eval_dset \
    --model_name gpt2-xl \
    --output_folder data/fact_recall_detection \
    --data_path "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/data_creation/final_splits/confident_fact_recall_preds.jsonl" \
```

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

CUDA_VISIBLE_DEVICES=1, python -m debugpy --wait-for-client --listen 5678 -m pararel.fact_recall_detection.create_eval_dset \
    --model_name gpt2-xl \
    --output_folder data/fact_recall_detection \
    --data_path "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/data_creation/final_splits/confident_fact_recall_preds.jsonl" \
```

## 2. Run `alvis_script_causal_trace.sh` to get the TEs.



## 3. Then run `process_relation.sh` to get a nice dataset for analysis.

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m pararel.fact_recall_detection.process_data \
    --output_file /cephyr/users/lovhag/Alvis/projects/rome/data/fact_recall_detection/gpt2_xl_final.csv \
    --queries_data_file /cephyr/users/lovhag/Alvis/projects/rome/data/fact_recall_detection/gpt2_xl_preds.jsonl \
    --ct_data_folder /cephyr/users/lovhag/Alvis/projects/rome/data/fact_recall_detection/gpt2_xl_causal_trace \
```

## 4. Analyze this dataset in a notebook and compare to our fact recall detection annotations.

Use [pararel/fact_recall_detection/evaluation.ipynb](pararel/fact_recall_detection/evaluation.ipynb) (under development).