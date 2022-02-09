from collections import defaultdict
import os
import sys

sys.path.append(".")
sys.path.append("..")
from torch import nn
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import jsonlines
import pandas as pd
import torch
import argparse

key_dict = {
    "acronym": ("short", "long"),
    "definition": ("concept", "definition"),
    "context": ("concept", "context"),
}

# FIXME import error if import it from train.py
class DMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def write_res_for_manual_evaluation(concepts, sents, predicts, res_file):
    df = pd.DataFrame(columns=["concept", "sentences", "predicts"])
    df["concept"] = concepts
    df["sentences"] = sents
    df["predicts"] = predicts
    df.to_csv(res_file)


def select_sentence(concepts, sents, predicts, res_file, type, thrd=0.5):
    index = defaultdict(set)
    for c, s, p in zip(concepts, sents, predicts):
        if p > thrd:
            index[c].add((s, p))
    with jsonlines.open(res_file, "w") as fout:
        for c in index:
            k1, k2 = key_dict[type]
            index[c] = [
                x[0] for x in sorted(index[c], key=lambda x: x[1], reverse=True)
            ]
            fout.write({k1: c, k2: index[c]})


def test(outptu_dir, model, tokenizer, type):
    infile = os.path.join(outptu_dir, f"{type}.jsonl")
    eval_file = os.path.join(outptu_dir, f"{type}_eval.csv")
    sel_file = os.path.join(outptu_dir, f"{type}_sel.jsonl")
    k1, k2 = key_dict[type]
    concepts, sents = [], []
    with jsonlines.open(infile) as fout:
        for item in fout:
            for s in item[k2]:
                concepts.append(item[k1])
                sents.append(s)
    preds = run_prediction(sents, model, tokenizer)
    write_res_for_manual_evaluation(concepts, sents, preds, eval_file)
    select_sentence(concepts, sents, preds, sel_file, type)


def run_prediction(sents, model, tokenizer):
    encodings = tokenizer(
        sents,
        truncation=True,
        padding=True,
        max_length=128,
    )
    dataset = DMDataset(encodings, None)
    eval_dataloader = DataLoader(dataset, batch_size=8)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.eval()
    m = nn.Softmax(dim=-1)
    preds = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            scores = m(logits)[:, 1].tolist()
            preds.extend(scores)
    return preds


def run(proj_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    for direction in ["bot_up", "top_down"]:
        output_dir = os.path.join("./output", proj_name, direction)
        test(output_dir, model, tokenizer, "acronym")
        test(output_dir, model, tokenizer, "definition")
        test(output_dir, model, tokenizer, "context")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name")
    parser.add_argument("--model_path")
    args = parser.parse_args()
    run(args.proj_name, args.model_path)