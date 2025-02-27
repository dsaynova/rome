import os, json
import torch

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.8f')

from experiments.causal_trace import (
    ModelAndTokenizer
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    predict_from_input
)

torch.set_grad_enabled(False)

def read_jsonl_file(filename: str):
    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)
    return dataset

def reformat_pararel2rome(val, relation_id):
    template_val = val["pattern"].replace("[X]", "{}")
    template_val = template_val[:template_val.index(" [Y]")]
    
    return {"subject": val["sub_label"],
            "attribute": val["answers"][0],
            "template": template_val,
            "prediction": None,
            "prompt": template_val.replace("{}", val["sub_label"]),
            "relation_id": relation_id
           }
    
def template_ends_with_mask(template):
    return template["pattern"].replace(".","").strip()[-3:]=="[Y]"

def template_starts_with_subj(template):
    return template["pattern"].replace(".","").strip()[0:3]=="[X]"

def get_rank_of_pred(pred, model_probs):
    # assumes that pred is the token id and model_probs an array of dim (vocab_size,)
    sorted_indices = torch.argsort(model_probs, descending=True)
    return (sorted_indices==pred).nonzero()[0][0]

def predict_top_candidate_from_input(model, inp, candidates_tokens):
    out = model(**inp)["logits"]
    opt_probs = torch.softmax(out[-1], dim=1)[-1]
    candidate_opt_probs = opt_probs[candidates_tokens]
    p, preds_ix = torch.max(candidate_opt_probs, dim=0)
    preds = torch.index_select(candidates_tokens, 0, preds_ix)
    
    return preds, p, opt_probs

def main(model_name, relation, output_folder, pararel_data_path, cache_folder):
    print(f"Producing data for relation {relation}...")
    data_tmp = read_jsonl_file(os.path.join(pararel_data_path, f"{relation}.jsonl"))
    print(f"Data length including all templates: {len(data_tmp)}")    
    data_template = []
    for val in data_tmp:
        # only add samples for which the pattern is suitable for ARMs
        if template_ends_with_mask(val) and template_starts_with_subj(val):
            data_template.append(reformat_pararel2rome(val, relation))
    print(f"Data length after removing non-ARM-compatible and not subject-first templates: {len(data_template)}") 
    
    pararel_options_file = os.path.join(pararel_data_path, f"{relation}_options.txt")
    with open(pararel_options_file) as f:
        options_all = [line.strip() for line in f.readlines()]  
        
        
    mt = ModelAndTokenizer(
        model_name,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
        cache_folder=cache_folder,
    )
    
    options=[]
    for o in options_all:
        if len(mt.tokenizer.encode(o, add_special_tokens=False))==1:
            options.append(o)
        
    data=[]
    for val in data_template:
        if val["attribute"] in options:
            data.append(val)
        
    print(f"Data length after removing partial objects: {len(data)}")     

    if "LlamaTokenizer" in str(type(mt.tokenizer)):
        #don't add special tokens (<s>)
        candidates_tokens = make_inputs(mt.tokenizer, [option for option in options], add_special_tokens=False)["input_ids"].squeeze()
    else:
        candidates_tokens = make_inputs(mt.tokenizer, [" "+option for option in options])["input_ids"].squeeze()
    assert candidates_tokens.shape==(len(options),)
    token2id = {options[ix]: candidates_tokens[ix].item() for ix in range(len(options))}
    
    prompts = [val["prompt"] for val in data]
    attributes = [val["attribute"] for val in data]
    preds_list = []
    p_list = []
    candidate_preds_list = []
    candidate_p_list = []
    candidate_ranks_list = []
    correct_ranks_list = []
    correct_p_list = []
    lama_prompt_top10_tokens = {}
    top10_tokens_list = []
    top10_tokens_probs_list = []
    with torch.no_grad():
        for i in range(len(prompts)):
            inp = make_inputs(mt.tokenizer, [prompts[i]])
            
            preds, p = predict_from_input(mt.model, inp)
            preds_list.append(int(preds))
            p_list.append(float(p))
            
            candidate_preds, candidate_p, opt_probs = predict_top_candidate_from_input(mt.model, inp, candidates_tokens)
            candidate_ranks = get_rank_of_pred(candidate_preds, opt_probs)
            candidate_preds_list.append(int(candidate_preds))
            candidate_p_list.append(float(candidate_p))
            candidate_ranks_list.append(int(candidate_ranks))
            
            correct_ranks = get_rank_of_pred(token2id[attributes[i]], opt_probs)
            correct_p = opt_probs[token2id[attributes[i]]]
            correct_ranks_list.append(int(correct_ranks))
            correct_p_list.append(float(correct_p))
            
            if data[i]["subject"] in lama_prompt_top10_tokens:
                top10_tokens = lama_prompt_top10_tokens[data[i]["subject"]]
            else:
                _, top10_tokens = torch.topk(opt_probs, 10)
                top10_tokens = top10_tokens.tolist()
                lama_prompt_top10_tokens[data[i]["subject"]] = top10_tokens
            top10_tokens_probs = opt_probs[top10_tokens]
            top10_tokens_list.append(top10_tokens)
            top10_tokens_probs_list.append(top10_tokens_probs.tolist())
            
    answers = decode_tokens(mt.tokenizer, preds_list)
    candidate_answers = decode_tokens(mt.tokenizer, candidate_preds_list)
    top10_tokens = [decode_tokens(mt.tokenizer, val) for val in top10_tokens_list]
    
    printable_model_name = model_name.split("/")[-1].replace("-", "_")
    output_file = os.path.join(output_folder, f"{relation}_{printable_model_name}_preds.jsonl")
    f = open(output_file, "w")
    for i, d in enumerate(data):
        dict_results = d
        dict_results["known_id"] = i
        dict_results["prediction"] = answers[i]
        dict_results["prediction_p"] = round(p_list[i], 8)
        
        dict_results["candidate_prediction"] = candidate_answers[i]
        dict_results["candidate_p"] = round(candidate_p_list[i], 8)
        dict_results["candidate_rank"] = candidate_ranks_list[i]
        
        dict_results["gold_rank"] = correct_ranks_list[i]
        dict_results["gold_p"] = round(correct_p_list[i], 8)
        
        dict_results["top10_tokens"] = top10_tokens[i]
        dict_results["top10_tokens_probs"] = top10_tokens_probs_list[i]
        f.write(json.dumps(dict_results))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", choices=["gpt2-xl", "EleutherAI/gpt-j-6B", "meta-llama/Llama-2-7b-hf"]
    )
    parser.add_argument(
        "--relation", choices=["P101", "P103", "P106", "P127", "P131", "P136", "P1376", "P138", "P140", "P1412", "P159", 
                               "P17", "P176", "P178", "P19", "P20", "P264", "P27", "P276", "P279", "P30", "P36", "P361", 
                               "P364", "P407", "P413", "P449", "P495", "P740", "P937"]
    )
    parser.add_argument(
        "--output_folder", type=str
    )
    parser.add_argument(
        "--pararel_data_path", type=str, default="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space"
    )
    parser.add_argument(
        "--cache_folder", required=True)
    args = parser.parse_args()

    main(
        args.model_name,
        args.relation,
        args.output_folder,
        args.pararel_data_path,
        args.cache_folder
    )
