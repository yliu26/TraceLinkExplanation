from typing import Dict, List

from gensim.models import TfidfModel
import argparse
import pandas as pd
import os
import sys

from jsonlines import jsonlines

sys.path.append("../..")
from domain_data_collection.relation_graph import RelationGraph
from gensim.corpora import Dictionary
from concept_detection.EntityDetection import DomainKG
import json
from collections import Counter, defaultdict
from gensim import matutils
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import math as m

version = "2.2.1"


def read_project(dir_path):
    arts = pd.read_csv(os.path.join(dir_path, "artifacts.csv"))
    lks = pd.read_csv(os.path.join(dir_path, "links.csv"))

    # annotated part
    cpts = pd.read_csv(os.path.join(dir_path, "concepts.csv"))
    tokens = pd.read_csv(os.path.join(dir_path, "tokens.csv"))
    concept_dict, token_dict = {}, {}
    for idx, row in cpts.iterrows():
        concept_dict[row["ids"]] = eval(row["phrase"])
    for idx, row in tokens.iterrows():
        token_dict[row["id"]] = eval(row["tokens"])

    art_dict, link_dict = {}, defaultdict(set)
    for idx, row in arts.iterrows():
        art_dict[row["id"]] = row["arts"]
    for idx, row in lks.iterrows():
        link_dict[row["sid"]].add(row["tid"])
    return art_dict, link_dict, concept_dict, token_dict


def get_imp_score(concept, model, lemmatizer):
    tks = concept.split()
    score = 0
    for tk in tks:
        tk = lemmatizer.lemmatize(tk)
        score += model.idfs[model.id2word.token2id[tk]]
    return score / len(tks)


def get_link_score(stoken, ttoken, model):
    doc1_vec = model[model.id2word.doc2bow(stoken)]
    doc2_vec = model[model.id2word.doc2bow(ttoken)]
    score = matutils.cossim(doc1_vec, doc2_vec)
    return score


def get_lemmas(concept, lemmatizer):
    tks = concept.split()
    return [lemmatizer.lemmatize(x) for x in tks]


def get_relation(lm1, lm2):
    interset = lm1 & lm2
    if len(interset) == len(lm1) and len(interset) == len(lm2):
        return "same_as"
    elif len(interset) == len(lm1):
        return "parent_of"
    elif len(interset) == len(lm2):
        return "child_of"
    else:
        return None


def process_path(p):
    relation_list = []
    for i in range(1, len(p)):
        c1, c2 = p[i - 1], p[i]
        verb = rel_graph.g.get_edge_attribute_by_id(c1, c2, 0, "verb")
        if len(verb) > 0:
            verb = list(verb)[0]
        relation_list.append(
            {
                "concept_1": c1.replace(" ", "_"),
                "relation_type": verb,
                "concept_2": c2.replace(" ", "_"),
            }
        )
    return relation_list


def format_concept_list(words):
    return [x.replace(" ", "_") for x in words]


def format_concept_dict(word_dict):
    flat_dict = []
    for k, v in word_dict.items():
        flat_dict.append({k.replace(" ", "_"): v})
    return flat_dict


def is_same(q1, q2, lmtzr):
    lemm_token = lambda x: [lmtzr.lemmatize(t) for t in x.split()]
    return lemm_token(q1) == lemm_token(q2)


def main(
    arts: Dict,
    links: List,
    concepts: Dict,
    tokens: Dict,
    rel_graph: RelationGraph,
    vague_rel_dict,
    acr_index,
    def_index,
):
    lemmas = {}
    lemmatizer = WordNetLemmatizer()
    for doc in tokens:
        lemmas[doc] = [lemmatizer.lemmatize(token) for token in tokens[doc]]
    dct = Dictionary(lemmas.values())
    corpus = [dct.doc2bow(doc_token) for doc_token in lemmas.values()]
    model = TfidfModel(corpus, id2word=dct)

    ann_arts = {}

    visited = set()
    terminilogy_pool = set(rel_graph.g.vertices())
    max_imp = 0
    min_imp = m.inf
    for id in tqdm(arts, desc="prepare artifacts and relation graph"):
        text = arts[id]
        doc_concepts = set(concepts[id])
        terminilogy = set()  # terminlogy is the concept unique in domain
        definitions = {}
        importance_scores = {}
        for c in concepts[id]:
            c_lower = c.lower()
            importance_scores[c] = get_imp_score(c, model, lemmatizer)
            max_imp = max(max_imp, importance_scores[c])
            min_imp = min(min_imp, importance_scores[c])
            if c_lower in terminilogy_pool:
                terminilogy.add(c)
                if c in acr_index:
                    definitions[c] = acr_index[c]
                elif c in def_index:
                    definitions[c] = def_index[c][
                        0
                    ]  # todo add definition ranking method
            elif c_lower not in visited:
                visited.add(c_lower)
                c_lm = Counter(get_lemmas(c_lower, lemmatizer))
                rel_graph.add_vertex(c_lower)
                for t in rel_graph.g.vertices():
                    r = get_relation(c_lm, Counter(get_lemmas(t, lemmatizer)))
                    if r is not None:
                        rel_graph.add_relation((c_lower, r, t.lower()))
            c_mark = c.replace(" ", "_")
            text = text.replace(c, c_mark)

        # normalize importance score
        for c in importance_scores:
            importance_scores[c] = (importance_scores[c] - min_imp) / (
                max_imp - min_imp
            )

        del_c = [x for x in importance_scores if importance_scores[x] < 0.1]
        for c in del_c:
            if c in doc_concepts:
                doc_concepts.remove(c)
            if c in terminilogy:
                terminilogy.remove(c)
            if c in definitions:
                del definitions[c]
            if c in importance_scores:
                del importance_scores[c]

        ann_arts[id] = {
            "id": id,
            "text": text,
            "concepts": format_concept_list(doc_concepts),
            "terminilogy": format_concept_list(terminilogy),
            "definitions": format_concept_dict(definitions),
            "importance_scores": format_concept_dict(importance_scores),
        }

    rel_graph.dump("../")
    res = []
    max_score = 0
    min_score = m.inf
    for sid in tqdm(links, desc="process link"):
        if sid not in ann_arts:
            continue
        sart = ann_arts[sid]
        sart["query"] = "source"
        targets = []
        for tid in links[sid]:
            if tid not in ann_arts:
                continue
            tart = ann_arts[tid]
            tart["query"] = "target"
            # get links
            clink = []
            for sc in sart["concepts"]:
                for tc in tart["concepts"]:
                    squery = sc.lower().replace("_", " ")
                    tquery = tc.lower().replace("_", " ")
                    relation_list = []
                    if rel_graph.g.is_reachable(squery, tquery):
                        p = rel_graph.g.shortest_paths(squery, tquery)[0]
                        relation_list = process_path(p)
                    elif rel_graph.g.is_reachable(tquery, squery):
                        p = rel_graph.g.shortest_paths(tquery, squery)[0]
                        relation_list = process_path(p)
                    cterm = set()
                    if len(relation_list) > 0:
                        clink.append(
                            {
                                "source": sc,
                                "target": tc,
                                "relationship": relation_list,
                            }
                        )
                        cterm.add(sc)
                        cterm.add(tc)

            tart["score"] = get_link_score(lemmas[sid], lemmas[tid], model)
            max_score = max(tart["score"], max_score)
            min_score = min(tart["score"], min_score)
            tart["links"] = clink
            tart["type"] = "Regulatory code"
            targets.append(tart.copy())
        ann_arts[sid]["type"] = "Requirements"  # fixme
        res.append({"source": [ann_arts[sid]], "targets": targets})
    # normalize the link score
    for r in res:
        for t in r["targets"]:
            t["score"] = round((t["score"] - min_score) / (max_score - min_score), 3)
            if t["score"] > 1:
                print(r)
    with open(f"./annotated_link_{version}.json", "w") as fout:
        json.dump(res, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare link file for visualization")
    parser.add_argument(
        "--project_dir", help="project directory contain the artifact and trace link"
    )
    parser.add_argument("--kg_dir", help="directory store the knowledge graph")
    args = parser.parse_args()
    arts, links, concepts, tokens = read_project(args.project_dir)
    rel_graph = RelationGraph()
    rel_graph.load(args.kg_dir)
    vague_rel = dict()
    vr_file = os.path.join(args.kg_dir, "vague_relations.jsonl")
    if os.path.isfile(vr_file):
        with jsonlines.open(vr_file) as fin:
            for obj in fin:
                l, vb, r = obj["left"], obj["verb"], obj["right"]
                vague_rel[(l, r)] = vb
    df_file = os.path.join(args.kg_dir, "definitions.jsonl")
    acr_file = os.path.join(args.kg_dir, "acronym.jsonl")
    definitions, acronyms = {}, {}
    if os.path.isdir(df_file):
        with jsonlines.open(df_file):
            for obj in fin:
                definitions[obj["concept"]] = obj["definition"]
    if os.path.isdir(acr_file):
        with jsonlines.open(acr_file) as fin:
            for obj in fin:
                acronyms[obj["short"]] = obj["long"]

    with open(os.path.join(args.project_dir, "basic_relation.json")) as fin:
        basic_r = json.load(fin)
        for lk in basic_r:
            for r in lk["relation"]:
                left, vb, right = r[0], r[1], r[2]
                rel_graph.add_relation((left, vb, right))
    for acr in acronyms:
        rel_graph.add_relation((acr, "synonym", acronyms[acr]))
        rel_graph.add_relation((acronyms[acr], "synonym", acr))

    main(arts, links, concepts, tokens, rel_graph, vague_rel, acronyms, definitions)
