import argparse
import json
import os
import re

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import KnownsDataset
from util.globals import DATA_DIR

from experiments.causal_trace import (
    ModelAndTokenizer,
    collect_embedding_std,
    collect_embedding_gaussian,
    collect_embedding_tdist,
    plot_trace_heatmap
)
from experiments.causal_trace_pararel import (
    calculate_hidden_flow
)

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
            "meta-llama/Llama-2-7b-hf",
        ],
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace_pararel")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--only_te", default=False, action='store_true')
    aa("--make_plots", default=False, type=bool)
    aa("--cache_folder", default=None, required=False)
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

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype, cache_folder=args.cache_folder)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        knowns = []
        with open(args.fact_file, "r") as f:
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
            print("Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in ["mlp"]:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/{known_id}_{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expects=knowledge["top10_tokens"],
                    kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                    te_flag=args.only_te, #speedup for total effect only
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if args.only_te: #speedup for total effect only
                continue
            if args.make_plots:
                plot_result = dict(numpy_result)
                plot_result["kind"] = kind
                pdfname = f'{pdf_dir}/{known_id}_{kind_suffix}.pdf'
                plot_trace_heatmap(plot_result, savepdf=pdfname)

if __name__ == "__main__":
    main()