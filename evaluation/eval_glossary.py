import sys
from eval_acronym import read_manual_acrn_eval
from eval_concept import read_concept_answer
from tqdm import tqdm

sys.path.append(".")
from evaluation import utils
import jsonlines
import os
import matplotlib.pyplot as plt
from matplotlib_venn import *


def read_glossary(file_path):
    cpt_set = set()
    with jsonlines.open(file_path) as fin:
        for item in fin:
            cpt = item.get("concepts", item.get("short", None))
            if cpt is not None:
                cpt_set.add(cpt)
    return cpt_set


def find_concpet_in_art(art_cpts, target_cpt):
    res = set()
    for tcpt in target_cpt:
        if tcpt in art_cpts:
            res.add(tcpt)
        else:
            for cpt in art_cpts:
                if utils.find_concept_in_text(cpt, tcpt) or utils.find_concept_in_text(
                    tcpt, cpt
                ):
                    res.add(tcpt)
    return res


def run(proj_name):
    data_dir = os.path.join("./data/projects", proj_name)
    s_art, t_art, link_dict, cpts_set = utils.read_project(data_dir)
    info = dict()
    for d in ["bot_up", "top_down"]:
        out_dir = os.path.join("./output", proj_name, d)
        eval_dir = os.path.join("./evaluation", proj_name, d)

        def_index = utils.read_definition(out_dir, file_name="definition_sel.jsonl")
        def_ans = read_concept_answer(eval_dir, "def_manual_eval.txt")
        def_index = set([x for x in def_index if x in def_ans])

        acrn_index = utils.read_acronym(out_dir, file_name="acronym_sel.jsonl")
        acrn_ans = read_manual_acrn_eval(eval_dir)
        acrn_index = set([x for x in acrn_index if x in acrn_ans])

        ctx_index = utils.read_context(out_dir, file_name="context_sel.jsonl")
        ctx_ans = read_concept_answer(eval_dir, "ctx_manual_eval.txt")
        ctx_index = set([x for x in ctx_index if x in ctx_ans])

        d_cpts = set()
        d_cpts.update(acrn_index)
        d_cpts.update(ctx_index)
        d_cpts.update(def_index)
        info[d] = d_cpts
    gls_cpts = set()
    gls_cpts.update(read_glossary(os.path.join(data_dir, "glossary_acronym.jsonl")))
    gls_cpts.update(read_glossary(os.path.join(data_dir, "glossary_definition.jsonl")))

    # gen numbers
    gls_in_art = find_concpet_in_art(cpts_set, gls_cpts)
    top_down = find_concpet_in_art(cpts_set, info["top_down"])
    bot_up = find_concpet_in_art(cpts_set, info["bot_up"])
    venn_data = {}
    venn_data["111"] = gls_in_art & top_down & bot_up
    venn_data["110"] = (top_down & bot_up) - venn_data["111"]
    venn_data["101"] = (top_down & gls_in_art) - venn_data["111"]
    venn_data["011"] = (bot_up & gls_in_art) - venn_data["111"]
    venn_data["100"] = top_down - bot_up - gls_in_art
    venn_data["010"] = bot_up - top_down - gls_in_art
    venn_data["001"] = gls_in_art - top_down - bot_up
    res = dict()
    for k in venn_data:
        res[k] = len(venn_data[k])
    draw_venn3(res, proj_name)


def draw_venn3(res, proj_name):
    out = venn3(
        subsets=res, set_labels=("Top-down", "Bottom-up", "Glossary"), alpha=0.5
    )
    venn3_circles(subsets=res, linestyle="solid")
    for text in out.set_labels:
        text.set_fontsize(16)
    for i in range(len(out.subset_labels)):
        if out.subset_labels[i] is not None:
            text.set_fontsize(16)
    plt.savefig(f"./figures/venn3_{proj_name}.png")
    plt.clf()


if __name__ == "__main__":
    for proj_name in tqdm(["CCHIT", "CM1", "PTC"]):
        res = run(proj_name)