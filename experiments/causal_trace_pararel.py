import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import KnownsDataset
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally

from experiments.causal_trace import *

def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="gpt2-xl",
        choices=[
            "gpt2-xl",
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
        ],
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace_pararel")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--te_flag", default=False, type=bool)
    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if "20b" in args.model_name else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        knowns = []
        with open(args.fact_file) as f:
            for line in f:
                loaded_example = json.loads(line)
                knowns.append(loaded_example)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    answer_type2expect_field = {"gold": "attribute", "candidate": "candidate_prediction", "any": "prediction"}
    answer_type2p_field = {"gold": "gold_p", "candidate": "candidate_p", "any": "prediction_p"}

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in None, "mlp", "attn":
            kind_suffix = f"_{kind}" if kind else ""
            for answer_type in "gold", "candidate", "any":
                filename = f"{result_dir}/{known_id}_{answer_type}{kind_suffix}.npz"
                if not os.path.isfile(filename):
                    expect = knowledge[answer_type2expect_field[answer_type]]
                    result = calculate_hidden_flow(
                        mt,
                        knowledge["prompt"],
                        knowledge["subject"],
                        expect=expect, #hack
                        kind=kind,
                        noise=noise_level,
                        uniform_noise=uniform_noise,
                        replace=args.replace,
                        base_score=knowledge[answer_type2p_field[answer_type]], #hack: we already have this in the (model specific) data
                        te_flag=args.te_flag, #speedup for total effect only
                    )
                    numpy_result = {
                        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in result.items()
                    }
                    numpy.savez(filename, **numpy_result)
                else:
                    numpy_result = numpy.load(filename, allow_pickle=True)
                if args.te_flag: #speedup for total effect only
                    continue
                plot_result = dict(numpy_result)
                plot_result["kind"] = kind
                pdfname = f'{pdf_dir}/{known_id}_{answer_type}{kind_suffix}.pdf'
                plot_trace_heatmap(plot_result, savepdf=pdfname)
            

def get_model_prob_for_token(model, inp, token_id):
    with torch.no_grad():
        out = model(**inp)["logits"]
        probs = torch.softmax(out[:, -1], dim=1)
    return probs[:,token_id]

def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
    base_score=None,
    te_flag=False,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    [answer_t] = mt.tokenizer.encode(" "+expect.strip())

    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    ).item()
    if te_flag: 
        return dict(
        scores=None,
        low_score=low_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        window=window,
        kind=kind or "",
        answer=expect, #hack
    )
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        window=window,
        kind=kind or "",
        answer=expect, #hack
    )


if __name__ == "__main__":
    main()
