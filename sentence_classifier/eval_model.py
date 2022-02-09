import sys

sys.path.append(".")
sys.path.append("..")
from sentence_classifier.predict import run_prediction
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import argparse
import os
import jsonlines

key_dict = {
    "acronym": ("short", "long"),
    "definition": ("concept", "definition"),
    "context": ("concept", "context"),
}


def run(proj_name, model_path, direction, type):
    kg_dir = os.path.join("./output/", proj_name, direction)
    out_dir = os.path.join("./sentence_clasifier/", proj_name, direction)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    infile = os.path.join(kg_dir, f"{type}.jsonl")
    sel_file = os.path.join(out_dir, f"{type}_selected.jsonl")
    origin_file = os.path.join(out_dir, f"{type}.jsonl")
    k1, k2 = key_dict[type]
    concepts, sents = [], []
    with jsonlines.open(infile) as fout:
        for item in fout:
            for s in item[k2]:
                concepts.append(item[k1])
                sents.append(s)
    preds = run_prediction(sents, model, tokenizer)

    sel_index, origin_index = dict(), dict()

    for c, s, p in zip(concepts, sents, preds):
        if c not in sel_index:
            sel_index[c] = set()
        origin_index[c] = s
        if p > 0.5:
            sel_index[c].add((s, p))

    with jsonlines.open(sel_file, "w") as fout:
        for c in sel_index:
            k1, k2 = key_dict[type]
            sel_index[c] = [
                x[0] for x in sorted(sel_index[c], key=lambda x: x[1], reverse=True)
            ][:1]
            fout.write({k1: c, k2: sel_index[c]})
    with jsonlines.open(origin_file, "w") as fout:
        for c in origin_index:
            fout.write({k1: c, k2: origin_index[c]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name")
    parser.add_argument("--model_path")
    args = parser.parse_args()
    for d in ["top_down", "bot_up"]:
        for type in ["acronym", "definition", "context"]:
            run(args.proj_name, args.model_path, d, type)