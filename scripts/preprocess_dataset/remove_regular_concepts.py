import argparse

import pandas as pd
from jsonlines import jsonlines
from tqdm import tqdm
import re


def get_concepts(concept_file):
    cpt_df = pd.read_csv(concept_file)
    cpts = set()
    for idx, row in cpt_df.iterrows():
        art_cpts = eval(row["phrase"])
        cpts.update(art_cpts)
    return cpts


def load_regular_concepts(concept_file, max_num=10 ** 7):
    cnt = 0
    cpts = set()
    print("loading regular concepts")
    with open(concept_file, encoding="utf8") as fin:
        for line in tqdm(fin):
            parts = line.split(",")
            cpts.add(parts[0].lower())
            cnt += 1
            if cnt > max_num and max_num > 0:
                break
    return cpts


def is_valid(cpt):
    blk_ch = {"?", "!"}
    for c in cpt:
        if c in blk_ch:
            return False
    
    
    if re.match("req\d+", cpt.lower()):
        return False

    for c in cpt:
        if c.isalpha():
            return True
    return False


def remove_regular_concpts(reg_cpt_csv, raw_cpt_csv, out_cpt):
    reg_cpts = load_regular_concepts(reg_cpt_csv)
    raw_cpt_df = pd.read_csv(raw_cpt_csv)
    all_rd_cpts = set()
    with jsonlines.open(out_cpt, "w") as fout:
        for idx, row in raw_cpt_df.iterrows():
            art_id = row["ids"]
            art_cpts = eval(row["phrase"])
            rd_cpts = set()
            blst_cpts = set()
            for cpt in art_cpts:
                if cpt.lower() not in reg_cpts and is_valid(cpt):
                    rd_cpts.add(cpt)
                    all_rd_cpts.add(cpt)
                else:
                    blst_cpts.add(cpt)
            fout.write(
                {
                    "art_id": art_id,
                    "origin_concepts": art_cpts,
                    "reduced_concepts": list(rd_cpts),
                    "removed_concepts": list(blst_cpts),
                }
            )
    out_cpt = out_cpt.replace(".jsonl", "_flat.txt")
    with open(out_cpt, "w") as fout:
        for c in all_rd_cpts:
            fout.write(f"{c}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regular_cpt_csv", help="csv file contains the regular concepts"
    )
    parser.add_argument(
        "--artifact_cpt_csv", help="csv file contains the concepts in each artifact"
    )
    parser.add_argument(
        "--output_cpt", help="jsonline file with filtered concepts in each artifact"
    )
    args = parser.parse_args()
    remove_regular_concpts(args.regular_cpt_csv, args.artifact_cpt_csv, args.output_cpt)
