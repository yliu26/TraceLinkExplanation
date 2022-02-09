from collections import defaultdict
from typing_extensions import get_args
import jsonlines
import os
from pattmatch import kmp
import sys
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")
from domain_data_collection.relation_graph import RelationGraph
import utils, argparse


def acronym_as_explain(acrns, content, acrn_index):
    res = set()
    for short in acrns:
        for long in acrn_index[short]:
            if utils.find_concept_in_text(content, long):
                res.add((short, long))
    return res


def read_manual_acrn_eval(dir_path, file_name="acronym_manual_eval.csv"):
    file_path = os.path.join(dir_path, file_name)
    answer = set()
    with open(file_path) as fin:
        for line in fin:
            items = line.split(",")
            acrn, label = items[0], items[1]
            if "1" in label:
                answer.add(acrn)
    return answer


# Total acronyms
# How many acronyms appeared in the artifacts
# Acronym ambiguity distribution
# Acronym as explaination absolute value
def evaluate_acronym(proj_dir, acrn_index, eval_res_dir, answer):
    if not os.path.isdir(eval_res_dir):
        os.makedirs(eval_res_dir)
    s_art, t_art, link_dict, cpts_set = utils.read_project(proj_dir)

    art_acrn = dict()
    art_acrn_gt = defaultdict(set)
    concept_dict = utils.read_concept_dict(proj_dir)
    for sid in tqdm(s_art, desc="scan source artifact"):
        art_acrn[sid] = utils.find_acronym_in_text(
            text=s_art[sid], acrn_index=acrn_index
        )
    for tid in tqdm(t_art, desc="scan target artifact"):
        art_acrn[tid] = utils.find_acronym_in_text(
            text=t_art[tid], acrn_index=acrn_index
        )

    for id in concept_dict:
        cpts = concept_dict[id]
        for cpt in cpts:
            if cpt.isupper():
                art_acrn_gt[id].add(cpt)

    has_acrn_cnt = 0
    acronym_in_art = set()
    for id in art_acrn_gt:
        if len(art_acrn_gt[id]) > 0:
            has_acrn_cnt += 1
            acronym_in_art.update(art_acrn_gt[id])

    acrn_has_long_names = acronym_in_art & acrn_index.keys()
    acrn_long_name_file = os.path.join(eval_res_dir, "acrn_in_art_has_long_name.txt")
    true_acrn, false_acrn = 0, 0
    with open(acrn_long_name_file, "w") as fout:
        for a in acrn_has_long_names:
            fout.write(f"{a}:{acrn_index[a]}\n")
            if a in answer:
                true_acrn += 1
            else:
                false_acrn += 1

    stat = {
        "how many acronym are extracted from corpus": len(acrn_index),
        "how mnay acronym are detected in artifacts": len(acronym_in_art),
        "How many acronym detected in artifact have long names": len(
            acronym_in_art.intersection(acrn_index.keys())
        ),
        "how many artifacts": len(s_art) + len(t_art),
        "how many acronyms find in artifacts are correct":true_acrn,
        "how many acronyms find in artifacts are incorrect": false_acrn,
    }
    utils.write_dict(stat, os.path.join(e, "acronym_stat.txt"))
    return acronym_in_art


def eval_overlap(td, bu, acronym_in_art, out_file):
    td_unique = (set(td.keys()) - set(bu.keys())) & acronym_in_art
    bu_unique = (set(bu.keys()) - set(td.keys())) & acronym_in_art
    common = set(td.keys()) & set(bu.keys()) & acronym_in_art
    with open(out_file, "w") as fout:
        fout.write(f"top_down unique:{len(td_unique)}\n")
        fout.write(f"bot_up unique:{len(bu_unique)}\n")
        fout.write(f"common: {len(common)}\n")


if __name__ == "__main__":
    proj_dir, out_dir, eval_dir = utils.get_args()

    top_down_acrn_index = utils.read_acronym(
        os.path.join(out_dir, "top_down"), file_name="acronym_sel.jsonl"
    )
    top_down_acrn_answer = read_manual_acrn_eval(os.path.join(eval_dir, "top_down"))
    bot_up_acrn_index = utils.read_acronym(
        os.path.join(out_dir, "bot_up"), file_name="acronym_sel.jsonl"
    )
    bot_up_acrn_answer = read_manual_acrn_eval(os.path.join(eval_dir, "bot_up"))

    both_acrn_index = dict()
    both_acrn_index.update(top_down_acrn_index)
    both_acrn_index.update(bot_up_acrn_index)

    both_acrn_answer = set()
    both_acrn_answer.update(top_down_acrn_answer)
    both_acrn_answer.update(bot_up_acrn_answer)

    eval_out = [os.path.join(eval_dir, x) for x in ["both", "top_down", "bot_up"]]
    arcn_list = [both_acrn_index, top_down_acrn_index, bot_up_acrn_index]
    answers = [both_acrn_answer, top_down_acrn_answer, bot_up_acrn_answer]
    for e, a, ans in zip(eval_out, arcn_list, answers):
        acronym_in_art = evaluate_acronym(proj_dir, a, e, ans)
        if e.endswith("both"):
            overlap_file = os.path.join(eval_dir, "both", "acrn_overlap.txt")
            eval_overlap(
                top_down_acrn_index, bot_up_acrn_index, acronym_in_art, overlap_file
            )
