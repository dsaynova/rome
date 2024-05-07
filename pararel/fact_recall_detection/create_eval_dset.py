import os, json
import torch

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.8f')

from experiments.causal_trace import (
    ModelAndTokenizer
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens
)
from pararel.create_pararel_dsets import (
    read_jsonl_file
)
from tqdm import tqdm

torch.set_grad_enabled(False)

def main(model_name, output_folder, data_path, cache_folder):
    print(f"Producing data from '{data_path}'...")
    data_tmp = read_jsonl_file(data_path)
    print(f"Data length including all templates: {len(data_tmp)}")
    data = []
    for val in data_tmp:
        data.append(reformat_data_to_rome(val))
        
    mt = ModelAndTokenizer(
        model_name,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
        cache_folder=cache_folder,
    )
    
    prompts = [val["prompt"] for val in data]
    top10_tokens_list = []
    top10_tokens_probs_list = []
    print("Collecting predictions...")
    with torch.no_grad():
        for i in tqdm(range(len(prompts))):
            inp = make_inputs(mt.tokenizer, [prompts[i]])
            out = mt.model(**inp)["logits"]
            opt_probs = torch.softmax(out[-1], dim=1)[-1]
            
            _, top10_tokens = torch.topk(opt_probs, 10)
            top10_tokens = top10_tokens.tolist()
            top10_tokens_probs = opt_probs[top10_tokens]
            top10_tokens_list.append(top10_tokens)
            top10_tokens_probs_list.append(top10_tokens_probs.tolist())
            
    top10_tokens = [decode_tokens(mt.tokenizer, val) for val in top10_tokens_list]
    
    printable_model_name = model_name.split("/")[-1].replace("-", "_")
    output_file = os.path.join(output_folder, f"{printable_model_name}_preds.jsonl")
    f = open(output_file, "w")
    print("Saving the results...")
    for i, d in enumerate(data):
        dict_results = d
        dict_results["known_id"] = i
        dict_results["top10_tokens"] = top10_tokens[i]
        dict_results["top10_tokens_probs"] = top10_tokens_probs_list[i]
        f.write(json.dumps(dict_results))
        f.write("\n")
    f.close()

def reformat_data_to_rome(val):
    template_val = val["template"].replace("[X]", "{}")
    template_val = template_val[:template_val.index(" [Y]")]
    
    return {"subject": val["sub_label"],
            "attribute": val["obj_label"],
            "template": template_val,
            "prediction": None,
            "prompt": val["prompt"],
            "relation_id": val["predicate_id"]
           }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"]
    )
    parser.add_argument(
        "--output_folder", type=str
    )
    parser.add_argument(
        "--data_path", type=str, required=True
    )
    parser.add_argument(
        "--cache_folder", default=None, required=False)
    args = parser.parse_args()

    main(
        args.model_name,
        args.output_folder,
        args.data_path,
        args.cache_folder
    )
