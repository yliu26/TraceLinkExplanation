"""
Find the concept relations with more basic rules
"""
import argparse
from pathlib import Path

from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
import sys
sys.path.append("..")
from domain_data_collection import utils
import json

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def get_lemmas(concept, lemmatizer):
    tks = concept.split()
    return [lemmatizer.lemmatize(x) for x in tks]


def get_stemmer(concept, stemmer):
    tks = concept.split()
    return [stemmer.stem(tk) for tk in tks]


def read_project(proj_dir):
    art_path = Path(proj_dir, "artifacts.csv")
    lk_path = Path(proj_dir, "links.csv")
    cpt_path = Path(proj_dir, "concepts.csv")
    arts = utils.load_arts(art_path)
    lk = utils.load_links(lk_path)
    cpts = utils.load_concpet(cpt_path)
    return arts, lk, cpts


def get_rel_type(p1, p2):
    tk1, tk2 = p1.split(), p2.split()
    stem1, stem2 = get_stemmer(p1, stemmer), get_stemmer(p2, stemmer)
    lm1, lm2 = get_lemmas(p1, lemmatizer), get_lemmas(p2, lemmatizer)
    stem_str1, stem_str2 = " ".join(stem1), " ".join(stem2)
    lemm_str1, lemm_str2 = " ".join(lm1), " ".join(lm2)
    if lemm_str1 == lemm_str2:
        return "same_as"

    if stem_str1 == stem_str2:
        return "synonym_of"

    if lemm_str1 in lemm_str2:
        return "parent_of"

    if lemm_str2 in lemm_str1:
        return "child_of"

    if lm1[-1] == lm2[-1]:
        return "sibling"

    syn1, syn2 = {}, {}

    for lm in lm1:
        syn1[lm] = {lm}
        for syn in wordnet.synsets(lm):
            for l in syn.lemmas():
                syn1[lm].add(l.name())
    for lm in lm2:
        syn2[lm] = {lm}
        for syn in wordnet.synsets(lm):
            for l in syn.lemmas():
                syn2[lm].add(l.name())
    reasons = set()
    for i, l1 in enumerate(lm1):
        s1 = stem1[i]
        t1 = tk1[i]
        for j, l2 in enumerate(lm2):
            s2 = stem2[j]
            t2 = tk2[j]
            if t1 == t2 or l1 == l2:
                reasons.add(f"{t1} same_as {t2}")
            elif s1 == s2:
                reasons.add(f"{t1} synonym_of {t2}")
            else:
                inter = syn1[l1].intersection(syn2[l2])
                if len(inter) > 0:
                    reasons.add(f"{t1} synonym_of {t2}")
    if len(reasons) > 0:
        return ",".join(reasons)
    return None


def find_concept_relation(plist1, plist2):
    relations = []
    for p1 in plist1:
        for p2 in plist2:
            rel_type = get_rel_type(p1, p2)
            if rel_type:
                relations.append((p1, rel_type, p2))
    return relations


def main(proj_dir, out_file):
    arts, lk, cpts = read_project(proj_dir)
    results = []
    for sid, tid in lk:
        if sid not in arts or tid not in arts:
            continue
        sart, tart = arts[sid], arts[tid]
        scpt, tcpt = cpts[sid], cpts[tid]
        relations = list(set(find_concept_relation(scpt, tcpt)))
        results.append({
            "sid": sid,
            "tid": tid,
            "sart": sart,
            "tart": tart,
            "relation": relations
        })
    with open(out_file, 'w') as fout:
        json.dump(results, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="find related concept in source and target artifacts"
    )
    parser.add_argument("--proj_dir", help="concepts extracted from project artifacts")
    parser.add_argument("--out_file", help="output the relation in json format")
    args = parser.parse_args()
    main(proj_dir=args.proj_dir, out_file=args.out_file)
