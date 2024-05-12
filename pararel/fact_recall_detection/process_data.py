import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from RQ1.process_data import (
    get_results_for_token_ix, 
    get_results_for_subject
)

def main(queries_data_file, ct_data_folder, output_file):
    queries = pd.DataFrame()
    with open(queries_data_file, "r") as f:
        for line in f:
            queries = queries.append(json.loads(line), ignore_index=True)
            
    print(f"Processing data from '{ct_data_folder}' for {len(queries.subject.unique())} subjects...")
    data = []

    for sample_ix in tqdm(range(len(queries))):
        for token_ix in range(10):
            data_entry = get_data_entry(ct_data_folder, queries.loc[sample_ix], token_ix)
            data.append(data_entry)
            
    data = pd.DataFrame(data)
    data.to_csv(output_file, index=False)
    print(f"Processed data has been saved to '{output_file}'.")
    return

def get_data_entry(results_folder, queries_entry, token_ix):
    try:
        filename = os.path.join(results_folder, f"cases/{int(queries_entry['known_id'])}__mlp.npz")
        results = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"Could not find file {filename}. Skipping...")
        return None
    answer = queries_entry["top10_tokens"][token_ix].strip()
    results = get_results_for_token_ix(results, token_ix, answer)
    results = get_results_for_subject(results)

    data_entry = {"subject": queries_entry.subject,
                 "template": queries_entry.template,
                 "pred": answer,
                 "pred_rank": token_ix,
                 "correct_answer": queries_entry.attribute,
                 "te": results["high_score"]-results["low_score"],
                }
    return data_entry

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str
    )
    parser.add_argument(
        "--queries_data_file", type=str
    )
    parser.add_argument(
        "--ct_data_folder", type=str
    )
    args = parser.parse_args()

    main(
        args.queries_data_file,
        args.ct_data_folder,
        args.output_file
    )