import argparse, sys, os
from collections import Counter
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd

sys.path.append("../..")
from concept_detection.EntityDetection import DomainKG
import heapq

logger = logging.getLogger(__name__)


def select_concepts(in_file, out_file, ratio):
    # select most frequent concepts (above ratio) with heap

    hp = []
    df = pd.read_csv(in_file)
    hp_size = df.size * ratio
    for idx, row in df.iterrows():
        c, f = row["concept"], row["freq"]
        heapq.heappush(hp, (f, c))
        if len(hp) > hp_size:
            heapq.heappop(hp)

    df = pd.DataFrame()
    df["concept"] = [x[1] for x in hp]
    df["freq"] = [x[0] for x in hp]
    df.to_csv(out_file, index=True)


def extract_webbase_concept(in_dir, out_file):
    concepts = Counter()
    for fname in tqdm(os.listdir(in_dir), desc="files"):
        if fname.endswith("possf2"):
            lines = (
                Path(os.path.join(in_dir, fname))
                .read_text(encoding="utf8")
                .splitlines()
            )
            for line in lines:
                tks = line.split()
                if len(tks) == 0:
                    continue
                words, pos = [], []

                for tk in tks:
                    try:
                        w, p = tk.split("_")
                        words.append(w)
                        pos.append(p)
                    except Exception:
                        pass

                cpts = DomainKG.extract_concepts(words, pos)
                for c in cpts:
                    concepts[c.text] += 1

    df = pd.DataFrame()
    df["concept"] = concepts.keys()
    df["freq"] = concepts.values()
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract regular concept and rank them by frequency from general corpus"
    )
    parser.add_argument("--data_dir", help="directory of general corpus")
    parser.add_argument("--out_file", help="output of concepts")
    parser.add_argument("--ratio", help="ratio of concepts to keep as regular concepts")
    parser.add_argument("--overwrite", default=True, help="output of concepts")

    args = parser.parse_args()
    tmp_file = os.path.join(os.path.dirname(args.out_file), "all_regular_concepts.csv")
    if not os.path.isfile(tmp_file) or args.overwrite:
        extract_webbase_concept(in_dir=args.data_dir, out_file=tmp_file)

    if not os.path.isfile(args.out_file) or args.overwrite:
        select_concepts(tmp_file, args.out_file, ratio=float(args.ratio))
