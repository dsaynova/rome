# README
Get CT results for exact fact recall samples and accurate recall samples.

1. Reformat the data to suit the format expected by the ROME code.

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m pararel.eval_on_fact_recall_set.reformat_to_rome \
    --srcfile "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/1000_exact.jsonl" \
    --outfile "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_exact.json" \

# for the acc any rank split
python -m pararel.eval_on_fact_recall_set.reformat_to_rome \
    --srcfile "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/1000_accurate.jsonl" \
    --outfile "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_accurate.json" \

# for the acc rank 0 split
python -m pararel.eval_on_fact_recall_set.reformat_to_rome \
    --srcfile "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/1000_accurate_rank_0.jsonl" \
    --outfile "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_accurate_rank_0.json" \
```

2. Run the ROME code

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh) for exact recall.

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_acc.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh) for accurate samples (any rank).

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_acc_rank_0.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_rank_0.sh) for accurate samples (rank 0).

Debug
```bash
CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m pararel.eval_on_fact_recall_set.causal_trace \
    --model_name "gpt2-xl" \
    --fact_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/fact_recall_set.json" \
    --output_dir "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_test"
```

**Exact fact recall save**

The results for the 1000 exact recall samples have been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2377617`. (could also have been subsampled from the full recall save)

**Accurate samples save**

The results for 1000 accurate samples have been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_acc_2379451`.

**Accurate rank 0 samples save**

The results for 1000 accurate samples have been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_acc_rank_0_2379452`.

**Full recall save**

Including non-popular subjects. The results have currently been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2344981`. It took about 7 hours to run. Noise level used: 0.136.

**Old accurate samples save**

The results for 996 (4 had to be removed due to tokenizer issues) accurate samples have been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2377618`.

The problematic samples (consisting of several subtokens that don't work with the substring finding method) were removed from the data. These were the entries with known_id 562, 564, 623 and 938 in the (old) accuracy set.