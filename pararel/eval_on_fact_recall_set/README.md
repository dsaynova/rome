# README
Get CT results for exact fact recall samples and accurate recall samples.

## 1. Reformat the data to suit the format expected by the ROME code.

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

# for the random guesswork split
python -m pararel.eval_on_fact_recall_set.reformat_to_rome \
    --srcfile "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/1000_guesswork.jsonl" \
    --outfile "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_guesswork.json" \
```

## 2. Run the ROME code

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh) for exact recall.

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_acc.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing.sh) for accurate samples (any rank).

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_acc_rank_0.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_rank_0.sh) for accurate samples (rank 0).

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_guesswork.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_guesswork.sh) for random guesswork samples.

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

**Random guesswork save**

The results for 1000 accurate samples have been saved to `TBD`.

**Full recall save**

Including non-popular subjects. The results have currently been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2344981`. It took about 7 hours to run. Noise level used: 0.136.

**Old accurate samples save**

The results for 996 (4 had to be removed due to tokenizer issues) accurate samples have been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2377618`.

The problematic samples (consisting of several subtokens that don't work with the substring finding method) were removed from the data. These were the entries with known_id 562, 564, 623 and 938 in the (old) accuracy set.

## 3. Get the low/high confidence splits for the known samples
Use [pararel/eval_on_fact_recall_set/split_knowns_by_confidence.ipynb](pararel/eval_on_fact_recall_set/split_knowns_by_confidence.ipynb) for this.

## 4. Make plots

Use the code below:

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

# Exact recall
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_exact.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2377617/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/exact_recall" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Accurate
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_accurate.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_acc_2379451/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/accurate" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Accurate rank 0
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_accurate_rank_0.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_acc_rank_0_2379452/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/accurate_rank_0" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Random guesswork 
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_guesswork.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_guesswork_2388522/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/random_guesswork" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Knowns
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/known_1000.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/results/gpt2-xl/known_1000/causal_trace_1907775/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/knowns" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Synthetic prompt bias
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/prompt_bias_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/prompt_bias_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/prompt_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/prompt_bias_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/prompt_bias_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/prompt_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2-7B" \
    --archname "Llama-2-7B" \

# Synthetic person name bias
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/person_name_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/person_name_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/person_name_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Synthetic string match bias
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/string_match_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/string_match_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/string_match_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Generic LM
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/generic_samples/generic_samples.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/generic_samples/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/generic_samples" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Knowns low confidence (<0.2, 501 samples)
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_knowns_low_confidence.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/results/gpt2-xl/known_1000/causal_trace_1907775/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/knowns_low_confidence" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Knowns high confidence (>=0.2, 499 samples)
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_knowns_high_confidence.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/results/gpt2-xl/known_1000/causal_trace_1907775/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/knowns_high_confidence" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

```

## 5. Get special plots that are the combination of existing query files and CT results for different prediction mechanisms

This dataset combines the exact recall, biased recall and random guesswork samples equally.

1. Produce the dataset with the mixed samples using [get_combined_mechanisms_data.ipynb](get_combined_mechanisms_data.ipynb).
2. Get the plots for the dataset using the code below.

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_combined_mechanisms.json" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/combined_mechanisms" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \
```