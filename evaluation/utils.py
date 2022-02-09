import argparse
from jsonlines import jsonlines
import pandas as pd
import os
from collections import defaultdict
import random
from pattmatch import kmp
import pandas as pd


def read_project(dir_path):
    sarts = pd.read_csv(os.path.join(dir_path, "source_artifacts.csv"))
    tarts = pd.read_csv(os.path.join(dir_path, "target_artifacts.csv"))
    lks = pd.read_csv(os.path.join(dir_path, "links.csv"))

    # annotated part
    tokens = pd.read_csv(os.path.join(dir_path, "tokens.csv"))
    token_dict = {}
    concept_set = set()
    with open(os.path.join(dir_path, "reduced_concepts_flat.txt")) as fin:
        for c in fin:
            concept_set.add(c.strip("\n\t\r "))

    for _, row in tokens.iterrows():
        token_dict[str(row["id"])] = eval(row["tokens"])

    s_art, t_art, link_dict = dict(), dict(), defaultdict(set)
    for _, row in sarts.iterrows():
        id = str(row["id"])
        if row["arts"] == row["arts"]:
            s_art[id] = " ".join(token_dict[id])
    for _, row in tarts.iterrows():
        id = str(row["id"])
        if row["arts"] == row["arts"]:
            t_art[id] = " ".join(token_dict[id])

    for _, row in lks.iterrows():
        sid, tid = str(row["sid"]), str(row["tid"])
        if sid in s_art and tid in t_art:
            link_dict[sid].add(tid)
    return s_art, t_art, link_dict, concept_set


def read_acronym(dir_path, file_name="acronym.jsonl"):
    acr_file = os.path.join(dir_path, file_name)
    acr_index = dict()
    with jsonlines.open(acr_file) as fin:
        for obj in fin:
            if obj["short"].islower():
                continue
            options = set()
            for long_arc in obj["long"]:
                if long_arc.lower() not in options:
                    acr_index[obj["short"]] = set()

                options.add(long_arc)
                acr_index[obj["short"]].add(long_arc)
    return acr_index


def read_definition(dir_path, file_name="definition.jsonl"):
    defs = dict()
    def_file = os.path.join(dir_path, file_name)
    with jsonlines.open(def_file) as fin:
        for o in fin:
            defs[o["concept"]] = o["definition"]
    return defs


def read_concept_dict(dir_path):
    concept_dict = dict()
    cpts = pd.read_csv(os.path.join(dir_path, "concepts.csv"))
    for _, row in cpts.iterrows():
        concept_dict[row["ids"]] = eval(row["phrase"])
    return concept_dict


def read_context(dir_path, file_name="context.jsonl"):
    context = dict()
    ctx_file = os.path.join(dir_path, file_name)
    with jsonlines.open(ctx_file) as fin:
        for o in fin:
            context[o["concept"]] = o["context"]
    return context


def read_relation(dir_path, rel_type="clear"):
    rels = defaultdict(defaultdict)
    rel_path = os.path.join(dir_path, f"{rel_type}_relation.jsonl")
    with jsonlines.open(rel_path) as fin:
        for o in fin:
            l, v, r = o["left"], o["verb"], o["right"]
            rels[l][r] = v
            rels[r][l] = v
    return rels


def read_corpus(dir_path):
    res = dict()
    bot_up_file = os.path.join(dir_path, "bot_up_corpus.jsonl")
    top_down_file = os.path.join(dir_path, "top_down_corpus.jsonl")
    if os.path.isfile(bot_up_file):
        with jsonlines.open(bot_up_file) as fin:
            for o in fin:
                res[o["query"]] = o["sentences"]
    if os.path.isfile(top_down_file):
        with jsonlines.open(top_down_file) as fin:
            for o in fin:
                res[o["query"]] = o["sentences"]
    return res


def find_concept_in_text(text, concept):
    lw_cpt_tokens = concept.lower().split()
    lw_text_tokens = text.lower().split()
    return len(kmp(lw_text_tokens, lw_cpt_tokens))


def find_acronym_in_text(text, acrn_index):
    res = set()
    text_tokens = text.split()
    for short in acrn_index:
        acrn_tokens = short.split()
        if len(kmp(text_tokens, acrn_tokens)):
            res.add(short)
    return res


def write_dict(dict_res, file_path):
    with open(file_path, "w") as fout:
        for k, v in dict_res.items():
            fout.write(f"{k}:{v}\n")


def sample_dict(d, file_path, k=20, col1="key", col2="value"):
    k = min(k, len(d.items()))
    selected = random.sample(list(d.items()), k)
    df = pd.DataFrame()
    for k, v in selected:
        df = df.append({col1: k, col2: v}, ignore_index=True)
    df.to_csv(file_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", default="CCHIT")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--eval_dir", default="./evaluation")

    args = parser.parse_args()
    proj_dir = os.path.join(args.data_dir, "projects", args.proj_name)
    out_dir = os.path.join(args.output_dir, args.proj_name)
    eval_dir = os.path.join(args.eval_dir, args.proj_name)
    return proj_dir, out_dir, eval_dir
