import argparse
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import math

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

from pararel.eval_on_fact_recall_set.generate_plots import (
    create_plots
)

def main(args):
    kind = "mlp"
    for f in os.listdir(args.query_folder):
        filepath = os.path.join(args.query_folder, f)
        if os.path.isfile(filepath):
            print()
            print()
            print(f"Creating plots for {filepath}...")
            data = pd.read_json(filepath, lines=filepath.endswith(".jsonl"))
            print(f"The data contains {len(data)} entries. Generating results for the first 1000...")
            print()
            data = data.iloc[:1000]
            count = len(data)
            
            savefolder = os.path.join(args.savefolder, f.split(".")[0])
            create_plots(data, kind, count, args.arch, args.archname, savefolder)
            
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--query_folder",
        required=True,
        type=str,
        help="Folder with files with combined queries to process, with corresponding CT results folders.",
    )
    argparser.add_argument(
        "--savefolder",
        required=True,
        type=str,
        help="Folder to save plot results to.",
    )
    argparser.add_argument(
        "--arch",
        required=True,
        type=str,
    )
    argparser.add_argument(
        "--archname",
        required=True,
        type=str,
    )
    args = argparser.parse_args()

    print(args)
    main(args)