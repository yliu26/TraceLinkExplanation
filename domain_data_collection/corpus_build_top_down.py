import gzip, os, json, argparse
import sys

sys.path.append(".")
sys.path.append("..")
from multiprocessing import Pool, Process, Queue
from domain_data_collection.corpus_build_bot_up import get_concepts
from cleantext import clean
from domain_data_collection.utils import clean_paragraph
from collections import defaultdict
import jsonlines
from pattmatch.kmp import kmp

CORP_ABSTR = "ABSTRACT"
CORP_FULL = "FULL"
PROG_FILE = "progress_file.txt"
OUT_FILE = "top_down_corpus.jsonl"
MERGED_OUT_FILE = "top_down_corpus_merged.jsonl"


def read_gorc_file(fpath, data_type=CORP_ABSTR):
    if fpath.endswith(".gz"):
        with gzip.open(fpath, "r") as f:
            for line in f:
                line = json.loads(line)
                if type(line).__name__ != "dict":
                    print(type(line))
                else:
                    if line["grobid_parse"] != None:
                        text_body = None
                        if (
                            line["grobid_parse"]["abstract"] != None
                            and data_type == CORP_ABSTR
                        ):
                            text_body = line["grobid_parse"]["abstract"]
                        elif (
                            line["grobid_parse"]["body_text"] != None
                            and data_type == CORP_FULL
                        ):
                            text_body = line["grobid_parse"]["body_text"]

                        if text_body is not None:
                            for body in text_body:
                                if body["text"] != None:
                                    text = body["text"]
                                    yield text


def sent_process_worker(input_q, output_q, concepts):
    """
    1. Open a zip file
    2. Select the sentences contains the concept
    3. Send them to output queue
    """
    while True:
        fpath = input_q.get()
        if fpath is None:
            break
        for text in read_gorc_file(fpath):
            par = clean(text).strip("\n\t\r ")
            sents = clean_paragraph(par)
            for s in sents:
                stokens = set(s.split())
                if len(stokens) < 5:
                    continue

                for query in concepts:
                    if query.lower() in s.lower():
                        idxs = kmp(s.lower().split(), query.lower().split())
                        if len(idxs) > 0:
                            output_q.put(
                                {
                                    "query": query,
                                    "sent": s,
                                }
                            )
        output_q.put({"file": fpath})


def start_sent_selector(concepts, fpath_list, output_q, process_num):
    input_q = Queue()
    for fpath in fpath_list:
        input_q.put(fpath)
    workers = []
    for _ in range(process_num):
        input_q.put(None)
        w = Process(
            target=sent_process_worker,
            args=(
                input_q,
                output_q,
                concepts,
            ),
        )
        w.start()
        workers.append(w)
    return workers


def res_collector_worker(output_q, out_file, ckpt_file, proc_num):
    done_proc_cnt = 0
    processed_files = set()
    query_sent_map = defaultdict(set)
    while True:
        item = output_q.get()
        if item is None:
            done_proc_cnt += 1
            if done_proc_cnt == proc_num:
                break
        else:
            if "file" in item:
                processed_files.add(item["file"])
                if len(processed_files) % 20 == 0:
                    print(f"{len(processed_files)} files have been processed")
                    write_checkpoint(
                        out_file, ckpt_file, processed_files, query_sent_map
                    )
                    query_sent_map = defaultdict(set)
            else:
                if len(query_sent_map[item["query"]]) <= 50:
                    query_sent_map[item["query"]].add(item["sent"])
    write_checkpoint(out_file, ckpt_file, processed_files, query_sent_map)
    print("finished")


def start_res_collector(output_q, out_file, ckpt_file, proc_num):
    w = Process(
        target=res_collector_worker,
        args=(output_q, out_file, ckpt_file, proc_num),
    )
    w.start()
    return w


def write_checkpoint(out_file, ckpt_file, processed_files, query_sent_map):
    with open(ckpt_file, "w") as fout:
        for pfile in processed_files:
            fout.write(f"{pfile}\n")

    with jsonlines.open(out_file, "a") as fout:
        for query in query_sent_map:
            sents = query_sent_map[query]
            fout.write(
                {
                    "query": query,
                    "sent_num": len(sents),
                    "sentences": list(sents),
                }
            )


def merge_corpus(corpus_file, out_file):
    query_sent_map = defaultdict(set)
    with jsonlines.open(corpus_file) as fin:
        for obj in fin:
            query, sents = obj["query"], obj["sentences"]
            query_sent_map[query].update(sents)

    with jsonlines.open(out_file, "w") as fout:
        for q in query_sent_map:
            fout.write(
                {
                    "query": q,
                    "sent_num": len(query_sent_map[q]),
                    "sentences": list(query_sent_map[q]),
                }
            )


def corpus_scan(concepts, corpus_dir, out_dir, proc_num, is_resume):
    output_q = Queue()
    ckpt_file = os.path.join(out_dir, PROG_FILE)
    out_file = os.path.join(out_dir, OUT_FILE)
    merged_out_file = os.path.join(out_dir, MERGED_OUT_FILE)

    g = os.walk(corpus_dir)
    visited_file, fpath_list = set(), set()
    if os.path.isfile(ckpt_file) and is_resume:
        with open(ckpt_file) as fin:
            visited_file.update(fin.read().splitlines())

    for root, dir_list, file_list in g:
        for fname in file_list:
            fpath = os.path.join(root, fname)
            if fpath not in visited_file:
                fpath_list.add(fpath)
    print(
        f"{len(visited_file)} has been processed, {len(fpath_list)} to process in total files."
    )
    sel_workers = start_sent_selector(concepts, fpath_list, output_q, proc_num)
    col_worker = start_res_collector(output_q, out_file, ckpt_file, proc_num)

    for p in sel_workers:
        p.join()
        output_q.put(None)
    col_worker.join()
    merge_corpus(out_file, merged_out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_dir",
        default="/afs/crc.nd.edu/group/dmsquare/vol5/Data-CiteExplainer/gorc",
        help="path to the coprus",
    )
    parser.add_argument(
        "--concept_file", help="path to the file store the concepts for every artifacts"
    )
    parser.add_argument(
        "--out_dir", help="output the collected corpus in the format of json"
    )
    parser.add_argument(
        "--is_resume",
        default=True,
        help="resume work by reading existing progress_file.txt",
    )
    parser.add_argument(
        "--proc_num",
        default=24,
        type=int,
        help="process number",
    )
    args = parser.parse_args()
    cpt_set = get_concepts(args.concept_file)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    corpus_scan(
        concepts=cpt_set,
        corpus_dir=args.corpus_dir,
        out_dir=args.out_dir,
        proc_num=args.proc_num,
        is_resume=args.is_resume,
    )
