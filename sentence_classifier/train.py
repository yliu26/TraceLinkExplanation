from transformers import TrainingArguments, AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer
import torch

from datasets import load_metric
import numpy as np
import sys

sys.path.append("..")
sys.path.append(".")
from evaluation import utils
from nltk.tokenize import sent_tokenize
import os
from sklearn.model_selection import train_test_split
import argparse

metric = load_metric("f1")
lm_name = "allenai/scibert_scivocab_uncased"


def read_training_data(project_name):
    def get_sent_for_proj(dir_path):
        s_art, t_art, link_dict, concept_set = utils.read_project(dir_path)
        sents = set()
        for sid in s_art:
            sents.update(sent_tokenize(s_art[sid]))
        for tid in t_art:
            sents.update(sent_tokenize(t_art[tid]))
        return sents

    proj_root = "./data/projects"
    sents, labels = [], []
    for pname in ["CCHIT", "CM1", "PTC"]:
        dir_path = os.path.join(proj_root, pname)
        proj_sents = get_sent_for_proj(dir_path)
        sents.extend(proj_sents)
        labels.extend([1 if pname == project_name else 0] * len(proj_sents))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sents, labels, test_size=0.2
    )

    return {
        "train": (train_texts, train_labels),
        "val": (val_texts, val_labels),
    }


def run(proj_name):
    raw_datas = read_training_data(proj_name)
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    datasets = dict()
    for part in raw_datas.keys():
        texts, labels = raw_datas[part]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
        )
        datasets[part] = DMDataset(encodings=encodings, labels=labels)
    train(proj_name, datasets["train"], datasets["val"])


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(proj_name, train_data, eval_data):
    training_args = TrainingArguments(
        output_dir=f"./sentence_classifier/{proj_name}",
        report_to="tensorboard",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
        load_best_model_at_end=True,
        save_strategy="epoch",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=3,
    )

    model = AutoModelForSequenceClassification.from_pretrained(lm_name, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name")
    args = parser.parse_args()
    run(args.proj_name)