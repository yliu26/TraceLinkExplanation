# how many relations are extracted
# how many links can be explained with the relations? How many are one hop and how many are two hop
# [manual/random sample] how is the quality of the links
import sys
from eval_glossary import find_concpet_in_art

sys.path.append(".")
sys.path.append("..")
from scripts.case_study.case_generator import gen_concpet_relation
from scripts.annotate_link.trace_link_annotation import process_path
from domain_data_collection.relation_graph import RelationGraph
from tqdm import tqdm
import utils
import os


def get_cpts_for_text(text, cpts_set):
    res = set()
    for cpt in cpts_set:
        if "SHALL" in cpt:
            continue
        if utils.find_concept_in_text(text, cpt):
            res.add(cpt)
    return res


def get_sent_with_concept(sent_list, cpt):
    res = set()
    for sent in sent_list:
        if utils.find_concept_in_text(sent, cpt):
            res.add(sent)
    return res


def find_match_for_cpt(tcpt, cpts_set):
    res = set()
    for cpt in cpts_set:
        if utils.find_concept_in_text(tcpt, cpt):
            res.add(tcpt)
    return res


def evaluate_relation(proj_dir, knowledge_dir, eval_dir):
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    s_art, t_art, link_dict, cpts_set = utils.read_project(proj_dir)
    simple_rel = 0

    clear_rels = utils.read_relation(knowledge_dir)
    vague_rels = utils.read_relation(knowledge_dir, rel_type="vague")
    acrn_index = utils.read_acronym(knowledge_dir)

    sel_acrn = find_concpet_in_art(cpts_set, acrn_index)
    for short in acrn_index.keys():
        if short in sel_acrn:
            for long in acrn_index[short]:
                clear_rels[short][long] = "acronym"

    rel_graph = RelationGraph()
    for l in clear_rels:
        for r in clear_rels[l]:
            rel_graph.add_relation((l, clear_rels[l][r], r))
    for l in vague_rels:
        for r in clear_rels[l]:
            rel_graph.add_relation((l, clear_rels[l][r], r))

    rel_set = dict()
    simple_rel = dict()
    clink = []
    for sid in tqdm(link_dict):
        for tid in link_dict[sid]:
            scpts = get_cpts_for_text(s_art[sid], cpts_set)
            tcpts = get_cpts_for_text(t_art[tid], cpts_set)
            clink.extend(gen_concpet_relation(rel_graph, scpts, tcpts))
    for p in clink:
        ps, pt = p["source"], p["target"]
        if p["type"] == "simple":
            simple_rel[ps, pt] = p
        else:
            rel_set[ps, pt] = p

    with open(os.path.join(eval_dir, "simple_links.txt"), "w") as fout:
        for p in simple_rel:
            fout.write(f"{simple_rel[p]}\n")
    with open(os.path.join(eval_dir, "relation_explain.txt"), "w") as fout:
        for p in rel_set:
            fout.write(f"{rel_set[p]}\n")
    return {
        "simple_rel": len(simple_rel),
        "How many unique clear relationship in trace links": len(rel_set),
    }


if __name__ == "__main__":
    proj_dir, out_dir, eval_dir = utils.get_args()
    proj_name = os.path.basename(proj_dir)

    for d in ["top_down", "bot_up"]:
        knowledge_dir = os.path.join(out_dir, d)
        eval_out = os.path.join(eval_dir, d)
        r = evaluate_relation(proj_dir, knowledge_dir, eval_out)
        utils.write_dict(r, os.path.join(eval_dir, "relation_stat.txt"))
