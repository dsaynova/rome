# README

## Set up environment

The repo requests PyTorch 1.10.2 and Python 3.9.7, while this is not available on Alvis. Instead we use the following:
```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
```

It implements Python 3.9.5 and PyTorch 1.11.0.

To then set up the packages use:

``` bash
python -m venv /mimer/NOBACKUP/groups/snic2021-23-309/envs/rome
source /mimer/NOBACKUP/groups/snic2021-23-309/envs/rome/bin/activate
pip install -r requirements.txt
pip install checklist==0.0.11 allennlp==2.9.0
```

I have here made sure to install two troublesome packages separately after the majority of the virtual environment is in place.  

## Test standard causal code

CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m experiments.causal_trace \
    --model_name "gpt2-xl" \
    --noise_level 0.1

## Create ParaRel based data

We can create a dataset based on ParaRel onto which we can perform causal tracing, like the `known_1000.json` dataset. To this dataset, we add some metrics, as in the example below. 

```
{"subject": "Yui Ichikawa", "attribute": "Tokyo", "template": "{} originated from", "prediction": " from", "prompt": "Yui Ichikawa originated from", "relation_id": "P19", "prediction_p": 0.50611538, "candidate_prediction": " Japan", "candidate_p": 0.00038193, "candidate_rank": 90, "gold_rank": 587, "gold_p": 4.284e-05}
```

The code performs the following steps to generate the data:
- Filter out templates that are suitable for the autoregressive modelling setup and only use these.
- Ensure that the answer alternatives are encoded to only one token by the model tokenizer.
- Generate the model predictions together with some additional metrics.
- The probability metrics are rounded to a floating point precision of 8.

To generate the dataset for relation P19, use the following code (note: it utilizes Lovisa's path to ParaRel):

```bash
python -m pararel.create_pararel_dsets --model_name gpt2-xl --relation P19 --output_folder data --pararel_data_path "/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space"
```

Currently, data generated using this code will be saved to `data`.

### Debug

Add `CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m` before your code and use your python function like a module.


## Run causal tracing on the ParaRel datasets

```bash
python -m experiments.causal_trace_pararel \
    --model_name "gpt2-xl" \
    --fact_file "/cephyr/users/lovhag/Alvis/projects/rome/data/P19_gpt2_xl_preds.jsonl" \
    --output_dir "/cephyr/users/lovhag/Alvis/projects/rome/data/results/gpt2-xl/P19/causal_trace_pararel_test"
```

To run as a job on Alvis, use `pararel/alvis_script_causal_tracing.sh`.

### Performance results

Running on a T4 GPU for four hours produces results for approximately 40 samples (for 3 targets: _any_, _candidate_ and _gold_). So running one sample takes approximately 6 minutes.

## Analyze KL divergences for the top10 tokens predicted with the LAMA prompt

To get the distributions over impactful model states for different prompts and output tokens, run the following code for the relations of interest:
- [create_pararel_dsets.py](pararel/create_pararel_dsets.py) using e.g. [alvis_script_create_data.sh](pararel/alvis_script_create_data.sh), and then
- [causal_trace_pararel.py](experiments/causal_trace_pararel.py) using e.g. [alvis_script_causal_tracing.sh](pararel/alvis_script_causal_tracing.sh).

Then, the KL divergences can be analyzed using [analyze_pararel_causal_tracing_results_top10.ipynb](notebooks/analyze_pararel_causal_tracing_results_top10.ipynb).

# RQ1

First, get the queries and corresponding causal tracing results for all relations of interest.

Then, process the data by running the script `RQ1/process_relation.sh <relation-to-process>`. Make sure that you have run the following commands before doing this:
```
module purge
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1 torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
source venv/bin/activate
```

After this, the results are processed using notebooks in `notebooks/acl_2024`.