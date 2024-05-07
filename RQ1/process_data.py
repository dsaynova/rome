import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def main(queries_data_file, ct_data_folder, output_file):
    queries = pd.DataFrame()
    with open(queries_data_file) as f:
        for line in f:
            queries = queries.append(json.loads(line), ignore_index=True)
            
    print(f"Processing data from '{ct_data_folder}' for {len(queries.subject.unique())} subjects...")
    data = []
    no_candidate_ix = []
    for subject in tqdm(queries.subject.unique()):
        sample_ixs = list(queries[queries.subject==subject].index)
        lama_ix = sample_ixs[0]
        
        token_ix = 0
        answer = queries.loc[lama_ix]["top10_tokens"][token_ix]
        
        cand_answer = queries.loc[lama_ix]["candidate_prediction"]
        if lama_ix not in no_candidate_ix:
            try:
                cand_token_ix = queries.loc[lama_ix]["top10_tokens"].index(cand_answer)
            except: 
                cand_token_ix = None
                no_candidate_ix.append(lama_ix)
            
        for sample_ix in sample_ixs[1:]:
            # for any top pred
            sample_answer = queries.loc[sample_ix]["prediction"]
            data_entry = get_data_entry(ct_data_folder, sample_ix, lama_ix, queries, token_ix, answer.strip(), sample_answer.strip())
            data_entry["pred_type"] = "any"
            data.append(data_entry)

            # for candidate top pred
            sample_answer = queries.loc[sample_ix]["candidate_prediction"]
            if cand_token_ix is None:
                continue
            data_entry = get_data_entry(ct_data_folder, sample_ix, lama_ix, queries, cand_token_ix, cand_answer.strip(), sample_answer.strip())
            data_entry["pred_type"] = "candidate"
            data.append(data_entry)
            
    data = pd.DataFrame(data)
    data.to_csv(output_file, index=False)
    print(f"Processed data has been saved to '{output_file}'.")
    
    print(f"Candidate not in top 10 predictions for {len(no_candidate_ix)} subjects corresponding to indeces:")
    print(no_candidate_ix)
    return
    
def get_cos_dist(lama_scores, scores):
    return 1-get_cos_sim(lama_scores, scores)

def get_cos_sim(lama_scores, scores):
    if not lama_scores.shape == scores.shape:
        print("Warning: Different shapes of CT results, skipping this sample...")
        return None
    lama_scores = np.nan_to_num(lama_scores/np.linalg.norm(lama_scores, axis=1)[:, np.newaxis])
    scores = np.nan_to_num(scores/np.linalg.norm(scores, axis=1)[:, np.newaxis])
    sim = np.multiply(lama_scores, scores).sum(axis=1)
    return sum(sim)/len(sim)

def get_results_for_token_ix(results, token_ix, answer_for_token):
    results = dict(results)
    if not results["scores"] == np.array(None):
        results["scores"] = results["scores"][:,:,token_ix]
    results["low_score"] = results["low_score"][token_ix]
    results["high_score"] = results["high_score"][token_ix]
    results["answer"] = answer_for_token #a bit hacky to get the answer as argument
    return results
    
def get_results_for_subject(results):
    # assumes that results already has been filtered by token_id
    if not results["scores"] == np.array(None):
        results["scores"] = results["scores"][results["subject_range"][0]:results["subject_range"][1],:]
    results["input_tokens"] = results["input_tokens"][results["subject_range"][0]:results["subject_range"][1]]
    results["input_ids"] = results["input_ids"][results["subject_range"][0]:results["subject_range"][1]]
    results["subject_range"] = np.array([0,len(results["input_tokens"])])
    return results

def get_data_entry(results_folder, sample_ix, lama_ix, queries, token_ix, answer, sample_answer):
    results = np.load(os.path.join(results_folder, f"cases/{sample_ix}_any_mlp.npz"), allow_pickle=True)
    results = get_results_for_token_ix(results, token_ix, answer)
    results = get_results_for_subject(results)
    
    results_lama = np.load(os.path.join(results_folder, f"cases/{lama_ix}_any_mlp.npz"), allow_pickle=True)
    results_lama = get_results_for_token_ix(results_lama, token_ix, answer)
    results_lama = get_results_for_subject(results_lama)
    
    correct_answer = queries.loc[lama_ix].attribute

    data_entry = {"sample_ix": sample_ix, 
                 "lama_ix": lama_ix,
                 "subject": queries.loc[sample_ix].subject,
                 "lama_template": queries.loc[lama_ix]["template"],
                 "sample_template": queries.loc[sample_ix]["template"],
                 "lama_answer": answer,
                 "sample_answer": sample_answer,
                 "correct_answer": correct_answer,
                 "lama_correct": answer==correct_answer,
                 "sample_correct": sample_answer==correct_answer,
                 "lama_te": results_lama["high_score"]-results_lama["low_score"],
                 "sample_te": results["high_score"]-results["low_score"],
                 "is_consistent": sample_answer==answer,
                 "pairwise_sim": get_cos_sim(results_lama["scores"], results["scores"]),
                 "subj_len": len(results["input_ids"]),
                 "token_ix": token_ix               
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