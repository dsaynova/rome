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
python create_pararel_dsets.py --model_name gpt2-xl --relation P19 --output_folder data --pararel_data_path "/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space"
```

Currently, data generated using this code will be saved to `data`.

### Debug

Add `CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m` before your code and use your python function like a module.