# README
Get CT results for exact fact recall samples and accurate recall samples.

## 1. Reformat the data to suit the format expected by the ROME code.

### GPT2-XL
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

### Llama2-7B
No need?

## 2. Run the ROME code

### GPT2-XL 
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

**Full recall save**

Including non-popular subjects. The results have currently been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2344981`. It took about 7 hours to run. Noise level used: 0.136.

**Old accurate samples save**

The results for 996 (4 had to be removed due to tokenizer issues) accurate samples have been saved to `/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2377618`.

The problematic samples (consisting of several subtokens that don't work with the substring finding method) were removed from the data. These were the entries with known_id 562, 564, 623 and 938 in the (old) accuracy set.

### Llama2-7B

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_exact_llama2_7B.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_exact_llama2_7B.sh) for exact recall.

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_acc_rank_0_llama2_7B.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_rank_0_llama2_7B.sh) for accurate samples (rank 0).

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_guesswork_llama2_7B.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_guesswork_llama2_7B.sh) for random guesswork samples.

### Llama2-13B

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_exact_llama2_13B.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_exact_llama2_13B.sh) for exact recall.

Use [pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_guesswork_llama2_13B.sh](pararel/eval_on_fact_recall_set/alvis_script_causal_tracing_guesswork_llama2_13B.sh) for random guesswork samples.

## 3. Get the low/high confidence splits for the known samples
Use [pararel/eval_on_fact_recall_set/split_knowns_by_confidence.ipynb](pararel/eval_on_fact_recall_set/split_knowns_by_confidence.ipynb) for this.

## 4. Make plots

### GPT2-XL and Llama2-7B
Use the code below:

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

# Exact recall
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_exact.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_2377617/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/exact_recall" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \
# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/llama2_7B/1000_exact.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/causal_trace_exact_2398046/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/exact_recall" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \
# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/llama2_13B/1000_exact.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/causal_trace_exact_2904371/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/exact_recall" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \


# Accurate
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_accurate.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_acc_2379451/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/accurate" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \


# Accurate rank 0
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_accurate_rank_0.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_acc_rank_0_2379452/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/accurate_rank_0" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/llama2_7B/1000_accurate_rank_0.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/causal_trace_acc_rank_0_2398047/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/accurate_rank_0" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \


# Random guesswork 
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_guesswork.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/causal_trace_guesswork_2388522/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/random_guesswork" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/llama2_7B/1000_guesswork.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/causal_trace_guesswork_2398048/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/random_guesswork" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \
# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_sensitivity_recall_eval_sets/llama2_13B/1000_guesswork.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/causal_trace_guesswork_2904686/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/random_guesswork" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \

# Knowns
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/known_1000.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/results/gpt2-xl/known_1000/causal_trace_1907775/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/knowns" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \


# Synthetic prompt bias
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/prompt_bias_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/prompt_bias_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/prompt_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/prompt_bias_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/prompt_bias_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/prompt_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \
    
# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/synthetic_data/prompt_bias_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/synthetic_data/prompt_bias_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/prompt_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \

# Synthetic person name bias
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/person_name_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/person_name_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/person_name_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/person_name_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/person_name_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/person_name_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \    

# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/synthetic_data/person_name_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/synthetic_data/person_name_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/person_name_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \ 

# Synthetic string match bias
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/string_match_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/synthetic_data/string_match_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/string_match_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/string_match_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/synthetic_data/string_match_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/string_match_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \    

# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/synthetic_data/string_match_bias.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/synthetic_data/string_match_bias/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/string_match_bias" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \    

# Generic LM
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/generic_samples/generic_samples.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/gpt2_xl/generic_samples/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/generic_samples" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/generic_samples/generic_samples.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_7B/generic_samples/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/generic_samples" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \  

# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/generic_samples/generic_samples.jsonl" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/fact-recall-detection/data/CT_results/llama2_13B/generic_samples/generic_samples/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/generic_samples" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \  

# Knowns low confidence (<0.2, 501 samples)
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_knowns_low_confidence.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/results/gpt2-xl/known_1000/causal_trace_1907775/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/knowns_low_confidence" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Knowns high confidence (>=0.2, 499 samples)
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_knowns_high_confidence.json" \
    --CT_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/results/gpt2-xl/known_1000/causal_trace_1907775/cases" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/knowns_high_confidence" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

```

## 5. Get special plots that are the combination of existing query files and CT results for different prediction mechanisms

We use this to produce two datasets and corresponding plots for each model.

### Combo
Dataset that combines the exact recall, biased recall and random guesswork samples equally.

1. Produce the dataset with the mixed samples using [get_combined_mechanisms_data.ipynb](get_combined_mechanisms_data.ipynb).
2. Get the plots for the dataset using the code below.

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_combined_mechanisms.json" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/combined_mechanisms" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/1000_combined_mechanisms.json" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/combined_mechanisms" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \

# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/1000_combined_mechanisms.json" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/combined_mechanisms" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \
```

### Bias combo
Dataset that combines prompt bias, person name bias and lexical overlap samples. 

1. Produce the dataset with the mixed samples using [get_combined_bias_mechanisms_data.ipynb](get_combined_bias_mechanisms_data.ipynb).
2. Get the plots for the dataset using the code below.

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/1000_combined_bias_mechanisms.json" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/gpt2-xl/summary_pdfs/biased_recall" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/1000_combined_bias_mechanisms.json" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_7B/summary_pdfs/biased_recall" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \

# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/1000_combined_bias_mechanisms.json" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/eval_on_fact_recall_set/llama2_13B/summary_pdfs/biased_recall" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \
```

## 6. Further experiments with singleton samples and different combinations of prediction mechanisms

1. Produce the datasets with the mixed samples using [pararel/eval_on_fact_recall_set/get_data_for_combination_experiments.ipynb](pararel/eval_on_fact_recall_set/get_data_for_combination_experiments.ipynb).
2. Get the plots for the dataset using the code below. This code will produce plots for all datafiles in the given folder.

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate

# GPT2 - singleton
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data_experiments \
    --query_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/singleton/gpt2_xl" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/singleton/gpt2_xl/summary_pdfs" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# GPT2 - combinations
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data_experiments \
    --query_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/combined/gpt2_xl" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/combined/gpt2_xl/summary_pdfs" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama - singleton
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data_experiments \
    --query_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/singleton/llama2_7B" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/singleton/llama2_7B/summary_pdfs" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \

# Llama - combinations
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data_experiments \
    --query_folder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/combined/llama2_7B" \
    --savefolder "/cephyr/users/lovhag/Alvis/projects/rome/data/prediction_mech_comb_experiments/combined/llama2_7B/summary_pdfs" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \
```
