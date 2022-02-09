import re

import pandas as pd
from nltk import sent_tokenize


def read_regular_concepts(regcpt_file):
    regcpt = set()
    with open(regcpt_file) as fin:
        for line in fin:
            cpt, cnt = line.split(',')[-2:]
            if int(cnt) > 10000:
                regcpt.add(cpt.lower())
            else:
                break
    return regcpt


def clean_paragraph(doc):
    # add space around "-"
    doc = re.sub("-", " - ", doc)
    # merge multiple space
    doc = re.sub("\s+", " ", doc)
    # remove brackets with empty or non-alphbabatic  content
    doc = re.sub("[\[<\()][^a-zA-Z]+[\]\)>]", "", doc)
    # remove <EMAIL> and <URL>
    doc = re.sub("(<EMAIL>|<URL>)", "", doc)
    # split it into sentences
    sents = sent_tokenize(doc)
    # for each sentence strip - and space
    res = []
    for s in sents:
        first_upper = -1
        for i, c in enumerate(s):
            if c.isalpha() and c.isupper():
                first_upper = i
                break
        cs = s
        if first_upper >= 0:
            cs = s[first_upper:]

        if len(cs.split()) > 3:
            res.append(cs)

    return [x.rstrip("\n\t\r -") for x in res]


def load_arts(file):
    df = pd.read_csv(file)
    arts = {}
    for id, content in zip(df['id'], df['arts']):
        arts[id] = content
    return arts


def load_links(file):
    df = pd.read_csv(file)
    links = []
    for sid, tid in zip(df['sid'], df['tid']):
        links.append((sid, tid))
    return links


def load_concpet(file):
    df = pd.read_csv(file)
    cpts = {}
    for id, phrases in zip(df['ids'], df['phrase']):
        plist = eval(phrases)
        cpts[id] = plist
    return cpts
