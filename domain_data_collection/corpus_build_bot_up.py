import argparse
import os
import ssl
import sys

sys.path.append(".")
sys.path.append("..")
from domain_data_collection.utils import clean_paragraph
import pathlib

ssl._create_default_https_context = ssl._create_unverified_context
from queue import Queue
from time import sleep, time

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from cleantext import clean
from search_engines import Bing, Google
import jsonlines
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from pattmatch.kmp import kmp

blacklist = [
    "[document]",
    "noscript",
    "header",
    "html",
    "meta",
    "head",
    "input",
    "script",
    "a",
    "style",
]
OUT_FILE = "bot_up_corpus.jsonl"


def get_concepts(concept_file):
    cpts = set()
    if concept_file.endswith(".jsonl"):
        with jsonlines.open(concept_file) as fin:
            for obj in fin:
                cpts.update(obj["reduced_concepts"])
    elif concept_file.endswith(".csv"):
        cpt_df = pd.read_csv(concept_file)
        for idx, row in cpt_df.iterrows():
            art_cpts = eval(row["phrase"])
            cpts.update(art_cpts)

    elif concept_file.endswith(".txt"):
        with open(concept_file) as fin:
            for line in fin:
                cpts.add(line.strip("\n\t\r "))

    extra_cpts = set()
    for cpt in cpts:
        tokens = cpt.split()
        if len(tokens) > 1:
            for tk in tokens:
                if len(tk) >= 3 and tk.isupper():
                    extra_cpts.add(tk)
    print(extra_cpts)

    print(f"loaded {len(cpts)} concepts and added {len(extra_cpts)}")
    cpts.update(extra_cpts)
    return cpts


def check_sent_quality(sent):
    stoken = set(sent.split())
    if len(stoken) < 5 or len(stoken) > 75:
        return False
    if sent.endswith("?") or sent.endswith("!"):
        return False
    return True


def scrap_worker(url, query, clean_lines):
    try:
        line_cnt_with_query = 0
        if url.endswith(".pdf") or url.endswith(".doc"):
            return
        header = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"}
        proj_root = pathlib.Path(__file__).parent.parent.resolve()
        ca_path = os.path.join(proj_root, "venv/Lib/site-packages/certifi/cacert.pem")
        if not os.path.isfile(ca_path):
            ca_path = os.path.join(
                proj_root, "venv/lib/python3.7/site-packages/certifi/cacert.pem"
            )
        html = requests.get(url, headers=header, timeout=6, verify=ca_path).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.find_all(text=True)
        output = ""

        for t in text:
            if t.parent.name not in blacklist and not isinstance(t, Comment):
                output += "{} ".format(t)

        output = [
            y for y in [x.strip("\n\t\r ") for x in output.split("\n")] if len(y) > 0
        ]

        for line in output:
            par = clean(line).strip("\n\t\r ")
            sents = clean_paragraph(par)
            for s in sents:
                if line_cnt_with_query >= 20 or not check_sent_quality(s):
                    break
                if query.lower() in s.lower():
                    idxs = kmp(s.lower().split(), query.lower().split())
                    if len(idxs) > 0:
                        line_cnt_with_query += 1
                        clean_lines.put(s)
    except Exception as e:
        print(e)


def scrap_concept(qcpt, domain, page_num, visited_link, engine="bing"):
    # scrap corpus with search engine and select sentences contains the given concept
    if engine == "google":
        engine = Google()
        query = f'"{qcpt}" in {domain}'
    else:
        engine = Bing()
        query = f"'{qcpt}' in {domain}"
    clean_lines = Queue()

    qres = engine.search(query, pages=page_num)
    links = [x for x in qres.links() if x not in visited_link and ".pdf" not in x][:50]
    visited_link.update(links)
    with ThreadPool(30) as p:
        p.starmap(scrap_worker, [(x, qcpt, clean_lines) for x in links])
    res = set()
    while not clean_lines.empty():
        res.add(clean_lines.get())
    return res


def domain_corpus_builder(concepts, out_dir, page_num, domain, interval):
    visited_cpt, visited_link = set(), set()
    out_file = os.path.join(out_dir, OUT_FILE)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if os.path.isfile(out_file):
        with jsonlines.open(out_file) as fin:
            for obj in fin:
                visited_cpt.add(obj["query"])

    i, total = 0, len(concepts)
    with jsonlines.open(out_file, "a") as fout:
        print("start")
        for cpt in tqdm(concepts):
            start = time()
            i += 1
            print(f"{i}/{total}: {cpt}")
            if cpt in visited_cpt:
                continue
            visited_cpt.add(cpt)
            clean_lines = scrap_concept(
                qcpt=cpt, domain=domain, page_num=page_num, visited_link=visited_link
            )
            fout.write(
                {
                    "query": cpt,
                    "sent_num": len(clean_lines),
                    "sentences": list(clean_lines),
                }
            )
            end = time()
            if end - start < 10:
                sleep(interval)


if __name__ == "__main__":
    "Collect related sentences from websites by providing it with concepts"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept_file", help="path to the file store the concepts for every artifacts"
    )
    parser.add_argument(
        "--out_dir", help="output the collected corpus in the format of json"
    )
    parser.add_argument(
        "--page_num", default=5, type=int, help="the number of pages in search engine"
    )
    parser.add_argument("--domain", help="the domain of the concepts")
    parser.add_argument("--query_interval", default=5, type=float)
    args = parser.parse_args()

    cpt_set = get_concepts(args.concept_file)
    domain_corpus_builder(
        concepts=cpt_set,
        out_dir=args.out_dir,
        page_num=args.page_num,
        domain=args.domain,
        interval=args.query_interval,
    )
