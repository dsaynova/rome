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

def get_rank_of_pred(pred, model_probs):
    # assumes that pred is the token id and model_probs an array of dim (vocab_size,)
    sorted_indices = torch.argsort(model_probs, descending=True)
    return (sorted_indices==pred).nonzero()[0][0]

def predict_top_candidate_from_input(model, inp, candidates_tokens):
    out = model(**inp)["logits"]
    opt_probs = torch.softmax(out[:, -1], dim=1)
    candidate_opt_probs = opt_probs[:,candidates_tokens]
    p, preds_ix = torch.max(candidate_opt_probs, dim=1)
    preds = torch.index_select(candidates_tokens, 0, preds_ix)
    
    return preds, p, opt_probs

def main(model_name, relation, output_folder, pararel_data_path):
    print(f"Producing data for relation {relation}...")
    data_tmp = read_jsonl_file(os.path.join(pararel_data_path, f"{relation}.jsonl"))
    print(f"Data length including all templates: {len(data_tmp)}")    
    data = []
    for val in data_tmp:
        # only add samples for which the pattern is suitable for ARMs
        if template_ends_with_mask(val):
            data.append(reformat_pararel2rome(val, relation))
    print(f"Data length after removing non-ARM-compatible templates: {len(data)}") 
    
    pararel_options_file = os.path.join(pararel_data_path, f"{relation}_options.txt")
    with open(pararel_options_file) as f:
        options = [line.strip() for line in f.readlines()]   
    
    mt = ModelAndTokenizer(
        model_name,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
    )
    
    candidates_tokens = make_inputs(mt.tokenizer, [" "+option for option in options])["input_ids"].squeeze()
    assert candidates_tokens.shape==(len(options),)
    token2id = {options[ix]: candidates_tokens[ix].item() for ix in range(len(options))}
    
    prompts = [val["prompt"] for val in data]
    attributes = [val["attribute"] for val in data]
    batchsize = 32
    preds_list = []
    p_list = []
    candidate_preds_list = []
    candidate_p_list = []
    candidate_ranks_list = []
    correct_ranks_list = []
    correct_p_list = []
    # TODO: get top candidate and its rank
    with torch.no_grad():
        for i in range(0, len(prompts), batchsize):
            inp = make_inputs(mt.tokenizer, prompts[i:i+batchsize])
            preds, p = predict_from_input(mt.model, inp)
            preds_list.extend(preds)
            p_list.extend(p)
            
            candidate_preds, candidate_p, opt_probs = predict_top_candidate_from_input(mt.model, inp, candidates_tokens)
            candidate_ranks = [get_rank_of_pred(candidate_preds[ix], opt_probs[ix]) for ix in range(opt_probs.shape[0])]
            candidate_preds_list.extend(candidate_preds)
            candidate_p_list.extend(candidate_p)
            candidate_ranks_list.extend(candidate_ranks)
            
            attributes_subset = attributes[i:i+batchsize]
            correct_ranks = [get_rank_of_pred(token2id[attributes_subset[ix]], opt_probs[ix]) for ix in range(opt_probs.shape[0])]
            correct_p = [opt_probs[ix, token2id[attributes_subset[ix]]] for ix in range(opt_probs.shape[0])]
            correct_ranks_list.extend(correct_ranks)
            correct_p_list.extend(correct_p)
    answers = decode_tokens(mt.tokenizer, preds_list)
    candidate_answers = decode_tokens(mt.tokenizer, candidate_preds_list)
    
    printable_model_name = model_name.split("/")[-1].replace("-", "_")
    output_file = os.path.join(output_folder, f"{relation}_{printable_model_name}_preds.jsonl")
    f = open(output_file, "w")
    for i, d in enumerate(data):
        dict_results = d
        dict_results["prediction"] = answers[i]
        dict_results["prediction_p"] = round(p_list[i].item(), 8)
        
        dict_results["candidate_prediction"] = candidate_answers[i]
        dict_results["candidate_p"] = round(candidate_p_list[i].item(), 8)
        dict_results["candidate_rank"] = candidate_ranks_list[i].item()
        
        dict_results["gold_rank"] = correct_ranks_list[i].item()
        dict_results["gold_p"] = round(correct_p_list[i].item(), 8)
        f.write(json.dumps(dict_results))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"]
    )
    parser.add_argument(
        "--relation", choices=["P19", "P101"]
    )
    parser.add_argument(
        "--output_folder", type=str
    )
    parser.add_argument(
        "--pararel_data_path", type=str, default="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space"
    )
    args = parser.parse_args()

    main(
        args.model_name,
        args.relation,
        args.output_folder,
        args.pararel_data_path
    )