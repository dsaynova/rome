# README

1. Reformat the data to suit the format expected by the ROME code.

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m pararel.eval_on_fact_recall_set.reformat_to_rome \
    --srcfile "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/data_creation/fact_recall_set.jsonl" \
    --outfile "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/fact_recall_set.json" \
```

2. Run the ROME code

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh).

Debug
```bash
CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m pararel.eval_on_fact_recall_set.causal_trace \
    --model_name "gpt2-xl" \
    --fact_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/fact_recall_set.json" \
    --output_dir "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_test"
```

The results have currently been saved to `/mimer/NOBACKUP/groups/snic2021-23-309/project-data/rome/logs/causal_tracing_gpt2_xl_fact_recall_set_2344981.out`. It took about 7 hours to run. Noise level used: 0.136.