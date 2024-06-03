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
    LINEPLOT_HIGH_SCORE, 
    LINEPLOT_LOW_SCORE,
    NORM_LINEPLOT_HIGH_SCORE,
    NORM_LINEPLOT_LOW_SCORE,
    Avg,
    plot_array,
    make_line_plot
)

def read_combined_knowlege(queries, do_normalize=False):
    (
        avg_fe,
        avg_ee,
        avg_le,
        avg_fa,
        avg_ea,
        avg_la,
        avg_hs,
        avg_ls,
        avg_fs,
        avg_fle,
        avg_fla,
    ) = [Avg() for _ in range(11)]
    for _, row in tqdm(queries.iterrows(), total=len(queries)):
        data = np.load(os.path.join(row.CT_results_dir, row.filename_template.format(row.known_id)))
        # old: Only consider cases where the model begins with the correct prediction
        if "correct_prediction" in data and not data["correct_prediction"]:
            raise ValueError("Data marked as 'not correct prediction'")
        scores = data["scores"].squeeze()
        if do_normalize:
            scores = (scores-data["low_score"].squeeze())/abs(data["high_score"].squeeze()-data["low_score"].squeeze())
        first_e, first_a = data["subject_range"]
        last_e = first_a - 1
        last_a = len(scores) - 1
        # original prediction
        avg_hs.add(data["high_score"].squeeze())
        # prediction after subject is corrupted
        avg_ls.add(data["low_score"].squeeze())
        avg_fs.add(scores.max())
        # some maximum computations
        avg_fle.add(scores[last_e].max())
        avg_fla.add(scores[last_a].max())
        # First subject middle, last subjet.
        avg_fe.add(scores[first_e])
        avg_ee.add_all(scores[first_e + 1 : last_e])
        avg_le.add(scores[last_e])
        # First after, middle after, last after
        avg_fa.add(scores[first_a])
        avg_ea.add_all(scores[first_a + 1 : last_a])
        avg_la.add(scores[last_a])

    result = np.stack(
        [
            avg_fe.avg(),
            avg_ee.avg(),
            avg_le.avg(),
            avg_fa.avg(),
            avg_ea.avg(),
            avg_la.avg(),
        ]
    )
    result_std = np.stack(
        [
            avg_fe.std(),
            avg_ee.std(),
            avg_le.std(),
            avg_fa.std(),
            avg_ea.std(),
            avg_la.std(),
        ]
    )
    print("Average Total Effect", avg_hs.avg() - avg_ls.avg())
    print(
        "Best average indirect effect on last subject",
        avg_le.avg().max() - avg_ls.avg(),
    )
    print(
        "Best average indirect effect on last token", avg_la.avg().max() - avg_ls.avg()
    )
    print("Average best-fixed score", avg_fs.avg())
    print("Average best-fixed on last subject token score", avg_fle.avg())
    print("Average best-fixed on last word score", avg_fla.avg())
    print("Argmax at last subject token", np.argmax(avg_le.avg()))
    print("Max at last subject token", np.max(avg_le.avg()))
    print("Argmax at last prompt token", np.argmax(avg_la.avg()))
    print("Max at last prompt token", np.max(avg_la.avg()))
    return dict(
        low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size()
    )

def main(args):
    kind = "mlp"
    data = pd.read_json(args.query_file, lines=args.query_file.endswith(".jsonl"))
    print(f"The data contains {len(data)} entries. Generating results for the first 1000...")
    print()
    data = data.iloc[:1000]
    count = len(data)
    
    print("Generating plots for CT results...")
    d = read_combined_knowlege(data)

    # get 2D heatmap    
    result = np.clip(d["result"] - d["low_score"], 0, None)
    plot_array(
        np.squeeze(result),
        kind=kind,
        title=None,
        low_score=0.0,
        high_score=None,
        archname=args.archname,
        savepdf=os.path.join(args.savefolder, "all_ranks.pdf"),
    )
    print("Average CT results heatmap saved!")
        
    # get lineplot without axis limits
    result = d["result"] - d["low_score"]
    result_std = d["result_std"]
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=None,
        high_score=None,
        count=count,
        archname=args.archname,
        savepdf=os.path.join(args.savefolder, "line_plot_all_ranks_free_lim.pdf"),
    )
    # with axis limits
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=LINEPLOT_LOW_SCORE,
        high_score=LINEPLOT_HIGH_SCORE,
        count=count,
        archname=args.archname,
        savepdf=os.path.join(args.savefolder, "line_plot_all_ranks.pdf"),
    )
    
    # get lineplot that measures direct probability
    result = d["result"]
    result_std = d["result_std"]
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=LINEPLOT_LOW_SCORE,
        high_score=LINEPLOT_HIGH_SCORE,
        count=count,
        archname=args.archname,
        savepdf=os.path.join(args.savefolder, "line_plot_prob_all_ranks.pdf"),
    )
    print("Average CT results line plot saved!")
    print()
    
    # normalized results
    print("Generating plots for normalized CT results...")
    norm_d = read_combined_knowlege(data, do_normalize=True)
    
    # get 2D heatmap
    plot_array(
        np.squeeze(norm_d["result"]),
        kind=kind,
        title=None,
        low_score=0.0,
        high_score=None,
        archname=args.archname,
        savepdf=os.path.join(args.savefolder, "all_ranks_norm.pdf"),
        cbar_title="NAIE"
    )
    print("Normalized average CT results heatmap saved!")
    
    # get lineplot without axis limits
    result = norm_d["result"]
    result_std = norm_d["result_std"]
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=None,
        high_score=None,
        count=count,
        archname=args.archname,
        savepdf=os.path.join(args.savefolder, "norm_line_plot_all_ranks_free_lim.pdf"),
        ylabel="Norm. average indirect effect on p(o)"
    )
    # with axis limits
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=NORM_LINEPLOT_LOW_SCORE,
        high_score=NORM_LINEPLOT_HIGH_SCORE,
        count=count,
        archname=args.archname,
        savepdf=os.path.join(args.savefolder, "norm_line_plot_all_ranks.pdf"),
        ylabel="Norm. average indirect effect on p(o)"
    )
    print("Normalized average CT results line plot saved!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--query_file",
        required=True,
        type=str,
        help="File with combined queries to process, with corresponding CT results folders.",
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