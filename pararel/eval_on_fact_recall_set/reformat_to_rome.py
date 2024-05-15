import argparse
import pandas as pd

def main(args):
    data = pd.read_json(args.srcfile, lines=True)
    data = data.rename(columns={"sub_label": "subject", "obj_label": "attribute", "answers": "prediction", "predicate_id": "relation_id"})
    data["known_id"] = data.index
    
    data.to_json(args.outfile, orient="records")
    print("Data reformatted and saved!")
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--srcfile",
        required=True,
        type=str,
        help="Source file from fact_recall_detection repo to reformat.",
    )
    argparser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="Filename to save processed data to.",
    )
    args = argparser.parse_args()

    print(args)
    main(args)