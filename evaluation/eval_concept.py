# how many concept are detected in artifacts
# how many concept have definitions
# how many concept have context
# [manual/random sample] how is the quality of the concept defintion/context

import utils
import os
import json


def evaluate_concept(proj_dir, def_index, ctx_index, eval_dir, def_ans, ctx_ans):
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    s_art, t_art, link_dict, cpts_set = utils.read_project(proj_dir)

    def_cnt = 0
    cpt_has_def, cpt_has_ctx = set(), set()
    for d in def_index:
        if d in cpts_set:
            def_cnt += 1
            cpt_has_def.add(d)
        else:
            for cpt in cpts_set:
                if utils.find_concept_in_text(cpt, d):
                    def_cnt += 1
                    cpt_has_def.add(d)
                    break
    ctx_cnt = 0
    for c in ctx_index:
        if c in cpts_set:
            ctx_cnt += 1
            cpt_has_ctx.add(c)
        else:
            for cpt in cpts_set:
                if utils.find_concept_in_text(cpt, c):
                    ctx_cnt += 1
                    cpt_has_ctx.add(c)
                    break

    file_has_cpt = 0
    arts_has_rich_cpt = set()
    for id in s_art:
        for c in cpts_set:
            if c in s_art[id]:
                file_has_cpt += 1
                if c in def_index or c in ctx_index:
                    arts_has_rich_cpt.add(id)

    for id in t_art:
        for c in cpts_set:
            if c in t_art[id]:
                file_has_cpt += 1
                if c in def_index or c in ctx_index:
                    arts_has_rich_cpt.add(id)

    link_total = 0
    link_has_rich_cpt = 0
    for sid in link_dict:
        for tid in link_dict[sid]:
            link_total += 1
            if sid in arts_has_rich_cpt or tid in arts_has_rich_cpt:
                link_has_rich_cpt += 1

    utils.sample_dict(def_index, os.path.join(eval_dir, "definition_sample.csv"))
    utils.sample_dict(ctx_index, os.path.join(eval_dir, "context_sample.csv"))
    correct_def = len(cpt_has_def & def_ans)
    incorrect_def = def_cnt - correct_def
    correct_ctx = len(cpt_has_ctx & ctx_ans)
    incorrect_ctx = ctx_cnt - correct_ctx
    stat = {
        "how many concepts are detected in artifacts": len(cpts_set),
        "how many concepts have definition": def_cnt,
        "how many concepts have correct def": correct_def,
        "how many concepts have incorrect def": incorrect_def,
        "how many concepts have correct ctx": correct_ctx,
        "how many concepts have incorrect ctx": incorrect_ctx,
        "how many concepts have context": ctx_cnt,
        "how many artifacts contains rich-concept (the concept have either definition or context)": f"{len(arts_has_rich_cpt)}/{file_has_cpt}",
        "how many links contains rich-concept explaination": f"{link_has_rich_cpt}/{link_total}",
    }
    utils.write_dict(stat, os.path.join(eval_dir, "concept_stat.txt"))


def read_concept_answer(dir, file_name):
    concept_ctx_file = os.path.join(dir, file_name)
    answers = set()
    with open(concept_ctx_file) as fin:
        for line in fin:
            items = line.split("\t")
            concept = json.loads(items[0])["concept"]
            if "1" in items[1]:
                answers.add(concept)
    return answers

if __name__ == "__main__":
    proj_dir, out_dir, eval_dir = utils.get_args()
    def_ans_file_name = "def_manual_eval.txt"
    ctx_ans_file_name = "ctx_manual_eval.txt"

    top_down_res_dir = os.path.join(eval_dir, "top_down")
    bot_up_res_dir = os.path.join(eval_dir, "bot_up")
    
    top_down_dir = os.path.join(out_dir, "top_down")
    top_down_def_index = utils.read_definition(
        top_down_dir, file_name="definition_sel.jsonl"
    )
    td_def_ans_index = read_concept_answer(top_down_res_dir, def_ans_file_name)
    td_ctx_ans_index = read_concept_answer(top_down_res_dir, ctx_ans_file_name)
    top_down_ctx_index = utils.read_context(top_down_dir, file_name="context_sel.jsonl")

    bot_up_dir = os.path.join(out_dir, "bot_up")
    bot_up_def_index = utils.read_definition(
        bot_up_dir, file_name="definition_sel.jsonl"
    )
    bu_def_ans_index = read_concept_answer(bot_up_res_dir, def_ans_file_name)
    bu_ctx_ans_index = read_concept_answer(bot_up_res_dir, ctx_ans_file_name)
    bot_up_ctx_index = utils.read_context(bot_up_dir, file_name="context_sel.jsonl")

    both_def_index = dict()
    both_def_index.update(top_down_def_index)
    both_def_index.update(bot_up_def_index)

    both_ctx_index = dict()
    both_ctx_index.update(top_down_ctx_index)
    both_ctx_index.update(bot_up_ctx_index)

    both_def_ans, both_ctx_ans = set(), set()
    both_def_ans.update(td_def_ans_index)
    both_def_ans.update(bu_def_ans_index)
    both_ctx_ans.update(td_ctx_ans_index)
    both_ctx_ans.update(bu_ctx_ans_index)

    eval_out = [os.path.join(eval_dir, x) for x in ["both", "top_down", "bot_up"]]
    def_list = [both_def_index, top_down_def_index, bot_up_def_index]
    def_answer = [both_def_ans, td_def_ans_index, bu_def_ans_index]
    ctx_answer = [both_ctx_ans, td_ctx_ans_index, bu_ctx_ans_index]
    ctx_list = [both_ctx_index, top_down_ctx_index, bot_up_ctx_index]

    for e, d, c, da, ca in zip(eval_out, def_list, ctx_list, def_answer, ctx_answer):
        print(e)
        r = evaluate_concept(proj_dir, d, c, e, da, ca)
