# Fact recall detection

Test the TE based approach on our created fact recall detection datasets.

Run our model on the queries and store corresponding TEs using the same approach as rome. 

## 1. Get model predictions for each query. (note that the code here should be different from our original - we do not wish to make pairwise comparisons here)

Run [pararel/fact_recall_detection/alvis_script_create_confident_eval_dset.sh](pararel/fact_recall_detection/alvis_script_create_confident_eval_dset.sh) and [pararel/fact_recall_detection/alvis_script_create_unconfident_eval_dset.sh](pararel/fact_recall_detection/alvis_script_create_unconfident_eval_dset.sh).

### Debug

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

## 2. Get the TEs

Run [pararel/fact_recall_detection/alvis_script_causal_trace_confident.sh](pararel/fact_recall_detection/alvis_script_causal_trace_confident.sh) and [pararel/fact_recall_detection/alvis_script_causal_trace_unconfident.sh](pararel/fact_recall_detection/alvis_script_causal_trace_unconfident.sh).


Remove the following entries for the unconfident split since the tokenizer couldn't deal with  the special tokens:
{"subject": "Rudolf K\u0159es\u0165an", "attribute": "Prague", "template": "{} was born in", "prediction": null, "prompt": "Rudolf K\u0159es\u0165an was born in", "relation_id": "P19", "known_id": 954, "top10_tokens": [" the", " 18", " Prague", " Poland", " Warsaw", " Vienna", " K", " Budapest", " Czech", " 1920"], "top10_tokens_probs": [0.06587336957454681, 0.06166274473071098, 0.0561046339571476, 0.029198696836829185, 0.020449575036764145, 0.017353605479002, 0.017239509150385857, 0.01592785306274891, 0.012018916197121143, 0.011126848869025707]}
{"subject": "Rudolf K\u0159es\u0165an", "attribute": "Prague", "template": "{} is originally from", "prediction": null, "prompt": "Rudolf K\u0159es\u0165an is originally from", "relation_id": "P19", "known_id": 955, "top10_tokens": [" Poland", " the", " Prague", " Czech", " Hungary", " Slovakia", " Warsaw", " Germany", " Austria", " Vienna"], "top10_tokens_probs": [0.16412849724292755, 0.12744970619678497, 0.06600821018218994, 0.04364611208438873, 0.03439946100115776, 0.030082937330007553, 0.02174679934978485, 0.02154010534286499, 0.017245547845959663, 0.01630954071879387]}
{"subject": "Jan \u010cul\u00edk", "attribute": "Prague", "template": "{} was born in", "prediction": null, "prompt": "Jan \u010cul\u00edk was born in", "relation_id": "P19", "known_id": 1077, "top10_tokens": [" the", " Prague", " Czech", " Hungary", " Poland", " K", " Budapest", " Z", " \ufffd", " \ufffd"], "top10_tokens_probs": [0.060881227254867554, 0.054137006402015686, 0.015723643824458122, 0.015623222105205059, 0.014262283220887184, 0.013862445019185543, 0.01355140469968319, 0.013518199324607849, 0.01091008447110653, 0.010314050130546093]}
{"subject": "Jan \u010cul\u00edk", "attribute": "Prague", "template": "{} is originally from", "prediction": null, "prompt": "Jan \u010cul\u00edk is originally from", "relation_id": "P19", "known_id": 1078, "top10_tokens": [" the", " Prague", " Poland", " Slovakia", " Croatia", " Hungary", " Czech", " Z", " \ufffd", " K"], "top10_tokens_probs": [0.11416597664356232, 0.09088738262653351, 0.05332690104842186, 0.049683578312397, 0.03691624477505684, 0.03417758271098137, 0.033805668354034424, 0.01830533891916275, 0.017992347478866577, 0.01742408610880375]}
{"subject": "Jan \u010cul\u00edk", "attribute": "Prague", "template": "{} was originally from", "prediction": null, "prompt": "Jan \u010cul\u00edk was originally from", "relation_id": "P19", "known_id": 1079, "top10_tokens": [" the", " Prague", " Poland", " Slovakia", " Czech", " Hungary", " Croatia", " Romania", " \ufffd", " \ufffd"], "top10_tokens_probs": [0.1552918404340744, 0.0445522777736187, 0.03948577493429184, 0.03758292272686958, 0.035635806620121, 0.031134523451328278, 0.027705511078238487, 0.018714137375354767, 0.01778777502477169, 0.016779165714979172]}

### Debug

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m pararel.fact_recall_detection.causal_trace \
    --model_name "gpt2-xl" \
    --fact_file "/cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_preds_debug.jsonl" \
    --output_dir "/cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_causal_trace_debug" \
    --only_te \
```

## 3. Process the resulting npz files

Run [pararel/fact_recall_detection/process_data.sh](pararel/fact_recall_detection/process_data.sh) to get a nice csv file for analysis:

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m pararel.fact_recall_detection.process_data \
    --output_file /cephyr/users/lovhag/Alvis/projects/rome/data/confident_fact_recall_detection/gpt2_xl_final.csv \
    --queries_data_file /cephyr/users/lovhag/Alvis/projects/rome/data/confident_fact_recall_detection/gpt2_xl_preds.jsonl \
    --ct_data_folder /cephyr/users/lovhag/Alvis/projects/rome/data/confident_fact_recall_detection/gpt2_xl_causal_trace \

python -m pararel.fact_recall_detection.process_data \
    --output_file /cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_final.csv \
    --queries_data_file /cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_preds.jsonl \
    --ct_data_folder /cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_causal_trace_2339146 \
```

### Debug

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m debugpy --wait-for-client --listen 5678 -m pararel.fact_recall_detection.process_data \
    --output_file /cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_final.csv \
    --queries_data_file /cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_preds.jsonl \
    --ct_data_folder /cephyr/users/lovhag/Alvis/projects/rome/data/unconfident_fact_recall_detection/gpt2_xl_causal_trace_2339146 \
```

## 4. Analyze the datasets in a notebook and compare to our fact recall detection sets

Use [pararel/fact_recall_detection/evaluation.ipynb](pararel/fact_recall_detection/evaluation.ipynb).