import argparse
import os
from collections import defaultdict
import sys
sys.path.append(".")
sys.path.append("..")

from jsonlines import jsonlines
from nltk.corpus import wordnet
from pattmatch import kmp
from tqdm import tqdm
from abbreviations import schwartz_hearst

from concept_detection.EntityDetection import DomainKG, Concept
from multiprocessing import Queue, Process

from domain_data_collection.utils import read_regular_concepts
ACRON_OUT = "acronym.jsonl"


def read_acronyms(acronym_file):
    acronym, inv_acronym = dict(), dict()
    with jsonlines.open(acronym_file) as fin:
        for obj in fin:
            acronym[obj["short"]] = obj["long"][0]
            inv_acronym[obj["long"][0]] = obj["short"]
    return acronym, inv_acronym


def write_acronyms(acronyms, out):
    with jsonlines.open(out, "w") as fout:
        for short in acronyms:
            fout.write({"short": short, "long": list(acronyms[short])})


# pip install git+git://github.com/tasos-py/Search-Engines-Scraper
# pip install abbreviations
# https://github.com/philgooch/abbreviation-extraction
def extract_acronym(clean_corpus, acronym_out):
    """
    Clean the corpus and keep the sentences related to the query concept
    :param raw_jl:
    :param out_file:
    :return:
    """
    with jsonlines.open(clean_corpus) as fin:
        acronyms = defaultdict(set)
        for obj in tqdm(fin):
            query = obj["query"]
            for s in obj["sentences"]:
                acronym_pairs = schwartz_hearst.extract_abbreviation_definition_pairs(
                    doc_text=s
                )
                for short, long in acronym_pairs.items():
                    if short.lower() != long.lower():
                        acronyms[short].add(long)
        write_acronyms(acronyms, acronym_out)


def get_valid_verb():
    # seed_verb = ["contain", "belong", "have", "include", "utilize", "use", "need", "require", "make", "conduct",
    #              "determine", "help", "achieve"]
    seed_verb = ["contain", "belong", "have", "include", "utilize", "use", "need", "determine"]
    verb_set = set()
    for sv in seed_verb:
        for syn in wordnet.synsets(sv):
            if syn.pos() != "v":
                continue
            verb_set.update([x.name().replace("_", " ") for x in syn.lemmas()])
    return verb_set


def is_valid_verb(verb, valid_verbs):
    extra_verb = {"such as", "of"}
    if verb in valid_verbs or verb in extra_verb:
        return True
    for tk in verb.split():
        if tk in valid_verbs:
            return True
    return False


def write_results(results, out_dir):
    def write_relation(rel_list, fout):
        for relation in rel_list:
            fout.write({"left": relation[0], "verb": relation[1], "right": relation[2]})

    disc_concepts = os.path.join(out_dir, "concept.jsonl")
    disc_vague_relation = os.path.join(out_dir, "vague_relation.jsonl")
    disc_clear_relation = os.path.join(out_dir, "clear_relation.jsonl")
    disc_definition = os.path.join(out_dir, "definition.jsonl")
    disc_context = os.path.join(out_dir, "context.jsonl")
    with jsonlines.open(disc_concepts, "w") as fout:
        for cpt in results["concepts"]:
            cpt = cpt.strip("\n\t\r ")
            fout.write({"concept": cpt})

    with jsonlines.open(disc_clear_relation, "w") as fout:
        write_relation(results["clear_relations"], fout)
    with jsonlines.open(disc_vague_relation, "w") as fout:
        write_relation(results["vague_relations"], fout)
    with jsonlines.open(disc_definition, "w") as fout:
        defs = results["definitions"]
        for cpt in defs:
            fout.write({"concept": cpt, "definition": list(defs[cpt])})
    with jsonlines.open(disc_context, "w") as fout:
        ctxs = results["context"]
        for cpt in ctxs:
            fout.write({"concept": cpt, "context": list(ctxs[cpt])})


def _worker_map(job_queue, out_queue, acronyms, inv_acronyms, reg_cpts):
    dkg = DomainKG()
    while True:
        is_def, is_ctx = False, False
        valid_verbs = get_valid_verb()
        clean_relation, vague_relation = set(), set()
        concepts = set()
        job = job_queue.get()

        if job is None:
            break
        s, query = job["sent"], job["query"]

        try:
            ann_sent = dkg.client.annotate(s).sentence[0]
        except:
            break

        type = is_definition(ann_sent, query, s)
        if type == "def":
            is_def = True
        elif type == "ctx":
            is_ctx = True

        words, pos = [], []
        for w in ann_sent.token:
            words.append(w.word)
            pos.append(w.pos)

        concept_index = dict()
        for c in dkg.extract_concepts(words, pos):
            for i in range(c.start, c.end):
                concept_index[i] = c

        # mark the query in the sentences to override the concept detection
        qtks = query.split()
        n = 0
        qconcept = []
        while n < len(words):
            match = True
            for k, qtk in enumerate(qtks):
                if n + k >= len(words) or words[n + k] != qtk:
                    match = False
                    break
            if match:
                for j in range(len(qtks)):
                    c = Concept(n, n + len(qtks), query)
                    concept_index[n + j] = c
                    qconcept.append(c)
                n += len(qtks) - 1
            n += 1

        # remove overlapped concept
        tmp_del = set()
        for idx in concept_index:
            c = concept_index[idx]
            for qc in qconcept:
                if (
                    qc.start <= c.start <= qc.end or qc.start <= c.end <= qc.end
                ) and not (qc.start == c.start and qc.end == c.end):
                    tmp_del.add(idx)
        for idx in tmp_del:
            del concept_index[idx]

        relations = dkg.extract_relations(ann_sent, concept_index)
        concepts.update([str(x[0]) for x in relations if is_valid_cpt(x[0], reg_cpts)])
        concepts.update([str(x[2]) for x in relations if is_valid_cpt(x[2], reg_cpts)])
        for x in relations:
            if not is_valid_cpt(x[0], reg_cpts) or not is_valid_cpt(x[2], reg_cpts):
                continue
            if is_valid_verb(x[1], valid_verbs):
                clean_relation.add(x)
            else:
                vague_relation.add(x)
        out_queue.put(
            {
                "query": query,
                "sent": s,
                "concepts": list(concepts),
                "is_def": is_def,
                "is_ctx": is_ctx,
                "clear_relations": clean_relation,
                "vague_relations": vague_relation,
            }
        )
    print("finished one process")
    out_queue.put(None)


def is_valid_cpt(cpt, reg_cpts):
    if cpt.lower() in reg_cpts:
        return False
    return True


def _worker_reduce(output_queue, map_num, out_dir):
    finished = 0
    concepts = set()
    definitions = defaultdict(set)
    context = defaultdict(set)
    clear_relation = set()
    vague_relation = set()

    while True:
        output = output_queue.get()
        if output is None:
            finished += 1
            if finished >= map_num:
                break
            else:
                continue

        concepts.update(output["concepts"])
        clear_relation.update(output["clear_relations"])
        vague_relation.update(output["vague_relations"])
        if output["is_def"]:
            definitions[output["query"]].add(output["sent"])
        elif output["is_ctx"]:
            context[output["query"]].add(output["sent"])

    print("finished collecting results")
    bup_res = {
        "concepts": list(concepts),
        "definitions": definitions,
        "context": context,
        "clear_relations": list(clear_relation),
        "vague_relations": list(vague_relation),
    }
    write_results(bup_res, out_dir)


def extract_definitions_and_relation(clean_corpus, acronym_file, out_dir, regcpt_file):
    reg_cpts = read_regular_concepts(regcpt_file)
    print("regular concept loaded...")
    acronyms, inv_acronyms = read_acronyms(acronym_file)
    map_num = 4
    mapworker = []
    job_q, out_q = Queue(), Queue()
    for _ in range(map_num):
        w = Process(
            target=_worker_map, args=(job_q, out_q, acronyms, inv_acronyms, reg_cpts)
        )
        mapworker.append(w)
        w.start()
    rp = Process(target=_worker_reduce, args=(out_q, map_num, out_dir))
    rp.start()

    with jsonlines.open(clean_corpus) as fin:
        for obj in fin:
            sents = obj["sentences"]
            query = obj["query"]
            for s in sents[:100]:
                job_q.put({"sent": s, "query": query})
    for w in mapworker:
        job_q.put(None)
    for w in mapworker:
        w.join()
    rp.join()
    job_q.close()
    out_q.close()


def is_definition(asent, query, s):
    if s.endswith("?") or s.endswith("!") or query.lower() not in s.lower():
        return False
    query_idxs = kmp([x.word.lower() for x in asent.token], query.lower().split())
    if len(query_idxs) == 0:
        return None
    query_idxs = query_idxs[0]
    fidx, lidx = query_idxs[0], query_idxs[-1]
    pre_tks, post_tks = asent.token[:fidx], asent.token[lidx:]
    if len(post_tks) > 0:
        if fidx < 2:
            in_deps, out_deps = defaultdict(dict), defaultdict(dict)
            for r in asent.enhancedPlusPlusDependencies.edge:
                t1_idx, rel, t2_idx = r.source - 1, r.dep.split(":"), r.target - 1
                rel = rel[0]

                out_rels = out_deps[t1_idx]
                tmp = out_rels.get(rel, [])
                tmp.append(t2_idx)
                out_deps[t1_idx][rel] = tmp

                in_rels = in_deps[t2_idx]
                tmp = in_rels.get(rel, [])
                tmp.append(t1_idx)
                in_deps[t2_idx][rel] = tmp

            for idx in range(query_idxs[0], query_idxs[-1]):
                subj_dep = in_deps[idx].get("nsubj", [])
                if len(subj_dep) > 0:
                    obj = subj_dep[0]
                    if (
                        asent.token[obj].pos.startswith("NN")
                        and len(out_deps[obj].get("cop", [])) > 0
                    ):
                        cop_idx = out_deps[obj]["cop"][0]
                        if asent.token[cop_idx].word in ["is", "are"] and (
                            fidx == 0 or asent.token[0].lemma in ["a", "an", "the"]
                        ):
                            return "def"
                        else:
                            return "ctx"
                    elif asent.token[obj].pos.startswith("VB") and asent.token[
                        obj
                    ].pos not in ["VBD"]:
                        return "ctx"
        return None


def extract_info(corpus, out_dir, regcpt_file):
    acronym_out = os.path.join(out_dir, ACRON_OUT)
    extract_acronym(clean_corpus=corpus, acronym_out=acronym_out)
    extract_definitions_and_relation(
        clean_corpus=corpus,
        acronym_file=acronym_out,
        out_dir=out_dir,
        regcpt_file=regcpt_file,
    )


if __name__ == "__main__":
    """
    create clean sentences for each query concept
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", help="corpus for extraction")
    parser.add_argument("--out_dir", help="dir to output the extracted information")
    parser.add_argument(
        "--regular_concepts", help="file to the list of regular concepts"
    )
    args = parser.parse_args()
    extract_info(
        corpus=args.corpus_file, out_dir=args.out_dir, regcpt_file=args.regular_concepts
    )
