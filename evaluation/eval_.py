import jsonlines
import pandas as pd
import os
from collections import defaultdict, Counter
from pattmatch import kmp
from tqdm import tqdm

from domain_data_collection.relation_graph import RelationGraph
import utils


def acronym_as_explain(s_acrns, t_acrns, acrn_index):
    res = set()
    t_longs = [x.lower() for x in t_acrns["long"]]
    for short in s_acrns["short"]:
        long = acrn_index[short].lower()
        if long in t_longs:
            res.add((short, long))
    return res


# Total acronyms
# How many acronyms appeared in the artifacts
# Acronym ambiguity distribution
# Acronym as explaination absolute value
def evaluate_acronym(proj_dir, output_dir, eval_res_dir="./eval_acrn"):
    if not os.path.isdir(eval_res_dir):
        os.makedirs(eval_res_dir)
    acrn_exp_file = os.path.join(eval_res_dir, "acrn_explain_file.jsonl")
    s_art, t_art, link_dict, concept_dict = utils.read_project(proj_dir)
    acrn_index = utils.read_acronym(output_dir)
    art_acrn = dict()
    for sid in s_art:
        art_acrn[sid] = utils.find_acronym_in_text(
            text=s_art[sid], acrn_index=acrn_index
        )
    for tid in t_art:
        art_acrn[tid] = utils.find_acronym_in_text(
            text=t_art[tid], acrn_index=acrn_index
        )

    exp_res = []
    for sid in link_dict:
        s_acrns = art_acrn[sid]
        for tid in link_dict[sid]:
            t_acrns = art_acrn[tid]

            explain = acronym_as_explain(s_acrns, t_acrns, acrn_index)
            explain.update(acronym_as_explain(t_acrns, s_acrns, acrn_index))
            if len(explain) > 0:
                exp_res.append({"sid": sid, "tid": tid, "explains": list(explain)})

    with jsonlines.open(acrn_exp_file, "w") as fout:
        for o in exp_res:
            fout.write(o)

    return {
        "acronym_num": len(acrn_index),  # Total acronyms
    }


def eval_definitions(sarts, tarts, links, kg_dir):
    def find_definition_in_arts(arts, defs, concepts):
        d_cnt, c_cnt = 0, 0
        for id in tqdm(arts, desc="definition eval"):
            content = arts[id]
            lw_tks = [x.strip("().,") for x in content.lower().split()]
            has_def, has_cpt = False, False

            for d in defs:
                if len(kmp(lw_tks, d.lower().split())) > 0:
                    has_def = True
                    break
            for c in concepts:
                if len(kmp(lw_tks, c.lower().split())) > 0:
                    has_cpt = True
                    break
            d_cnt += 1 if has_def else 0
            c_cnt += 1 if has_cpt else 0
        return d_cnt / len(arts), c_cnt / len(arts)

    def_eval_res = dict()
    defs, concepts = set(), set()
    def_file = os.path.join(kg_dir, "definition.jsonl")
    cpt_file = os.path.join(kg_dir, "concept.jsonl")
    with jsonlines.open(def_file) as fin:
        for o in fin:
            defs.add(o["concept"].lower())
    with jsonlines.open(cpt_file) as fin:
        for o in fin:
            concepts.add(o["concept"].lower())
    (
        def_eval_res["sart_contain_def_ratio"],
        def_eval_res["sart_contain_concept_ratio"],
    ) = find_definition_in_arts(sarts, defs, concepts)
    (
        def_eval_res["tart_contain_def_ratio"],
        def_eval_res["tart_contain_concept_ratio"],
    ) = find_definition_in_arts(tarts, defs, concepts)
    def_eval_res["concpet_has_definition"] = len(concepts.intersection(defs)) / len(
        concepts
    )
    return def_eval_res


def eval_clear_relation(sarts, tarts, links, concepts, kg_dir):
    general_concepts = {"information", "system", "ability", "results", "data", "time"}
    rel_graph = RelationGraph()
    rel_graph.load(kg_dir, link_file="clear_relation.jsonl")
    clear_rel_eval_res = dict()
    cpt_related_explain = Counter()
    debug_links = []
    for sid in tqdm(sarts, desc="process links"):
        for tid in tarts:
            if tid in links[sid]:
                label = True
            else:
                label = False
            has_related_concept = False
            for scpt in concepts[sid]:
                if scpt in general_concepts:
                    continue
                for tcpt in concepts[tid]:
                    if tcpt in general_concepts:
                        continue
                    if rel_graph.g.is_reachable(scpt, tcpt) or rel_graph.g.is_reachable(
                        tcpt, scpt
                    ):
                        has_related_concept = True
                        debug_links.append(
                            {"label": label, "s_concept": scpt, "t_concept": tcpt}
                        )
                        break
            if has_related_concept:
                cpt_related_explain[label] += 1
    with jsonlines.open("debug_clear_cpt_relation", "w") as fout:
        for o in debug_links:
            fout.write(o)
    clear_rel_eval_res["true_link_with_related_concepts"] = cpt_related_explain[True]
    clear_rel_eval_res["false_link_with_related_concepts"] = cpt_related_explain[False]
    return clear_rel_eval_res


def eval_vague_relation(sarts, tarts, links, concepts, kg_dir):
    general_concepts = {"information", "system", "ability", "results", "data", "time"}
    vague_relation = os.path.join(kg_dir, "vague_relation.jsonl")
    rel_set = set()
    debug_links = []
    with jsonlines.open(vague_relation) as fin:
        for o in fin:
            left, right = o["left"].lower(), o["right"].lower()
            rel_set.add((left, right))
    clear_rel_eval_res = dict()
    cpt_related_explain = Counter()
    for sid in tqdm(sarts, desc="process links"):
        for tid in tarts:
            if tid in links[sid]:
                label = True
            else:
                label = False
            has_related_concept = False
            for scpt in concepts[sid]:
                if scpt in general_concepts:
                    continue
                for tcpt in concepts[tid]:
                    if tcpt in general_concepts:
                        continue
                    if (scpt, tcpt) in rel_set or (tcpt, scpt) in rel_set:
                        has_related_concept = True
                        debug_links.append(
                            {"label": label, "s_concept": scpt, "t_concept": tcpt}
                        )
                        break

            if has_related_concept:
                cpt_related_explain[label] += 1
    with jsonlines.open("debug_vague_cpt_relation", "w") as fout:
        for o in debug_links:
            fout.write(o)
    clear_rel_eval_res["true_link_with_related_concepts"] = cpt_related_explain[True]
    clear_rel_eval_res["false_link_with_related_concepts"] = cpt_related_explain[False]
    return clear_rel_eval_res


if __name__ == "__main__":
    dir_path = "../data/projects/CCHIT"
    kg_dir = "../data/backup/www_BU/data/CCHIT/bup_res"
    eval_output_dir = "../data/backup/www_BU/eval_res"
    summary_file = os.path.join(eval_output_dir, "summary.jsonl")
    if not os.path.isdir(eval_output_dir):
        os.makedirs(eval_output_dir)
    s_art, t_art, links, concepts, tokens = read_project(dir_path)
    report = {}
    dataset_info = {
        "source #": len(s_art),
        "target #": len(t_art),
        "link #": len([y for x in links for y in links[x]]),
    }
    report["dataset_info"] = dataset_info
    print(dataset_info)

    acr_eval_res = evaluate_acronym(s_art, t_art, links, kg_dir)
    report["acronym_eval"] = acr_eval_res
    def_eval_res = eval_definitions(s_art, t_art, links, kg_dir)
    report["definition_eval"] = def_eval_res
    clear_rel_res = eval_clear_relation(s_art, t_art, links, concepts, kg_dir)
    report["clear_relation_eval"] = clear_rel_res
    print(clear_rel_res)
    vague_rel_res = eval_vague_relation(s_art, t_art, links, concepts, kg_dir)
    report["vague_relation_eval"] = vague_rel_res
    print(vague_rel_res)
    with jsonlines.open(summary_file, "w") as fout:
        fout.write(report)
