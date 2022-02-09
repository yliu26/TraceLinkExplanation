import os
import pathlib
import sys
import argparse
sys.path.append(".")
sys.path.append("..")

from concept_detection.EntityDetection import DomainKG
from scripts.preprocess_dataset.remove_regular_concepts import remove_regular_concpts

from concept_detection.DataReader import CCHITReader, PTCReader, CM1Reader, InfusionPumpReader
import pandas as pd

reader_map = {
    "CCHIT": CCHITReader,
    "PTC": PTCReader,
    "CM1": CM1Reader,
    # "InfusionPump": InfusionPumpReader
}


def concept_detect(proj_dir, sart_fname="source_artifacts.csv", tart_fname="target_artifacts.csv"):
    sart = pd.read_csv(os.path.join(proj_dir, sart_fname)).dropna()
    tart = pd.read_csv(os.path.join(proj_dir, tart_fname)).dropna()
    arts = sart["arts"].to_list() + tart["arts"].to_list()
    ids = sart["id"].to_list() + tart["id"].to_list()

    dkg = DomainKG()
    cpt_file = os.path.join(proj_dir, "concepts.csv")
    concepts, rels, tokens = dkg.build(arts)
    concept_df = pd.DataFrame()
    concept_df["phrase"] = concepts
    concept_df["ids"] = ids
    concept_df.to_csv(cpt_file)

    rel_df = pd.DataFrame()
    rel_df["id"] = ids
    rel_df["rels"] = rels
    rel_df.to_csv(os.path.join(proj_dir, "relations.csv"))

    tk_df = pd.DataFrame()
    tk_df["id"] = ids
    tk_df['tokens'] = tokens
    tk_df.to_csv(os.path.join(proj_dir, "tokens.csv"))
    return cpt_file


def preprocess(proj_dir, reg_cpt_file):
    reader = reader_map[os.path.basename(proj_dir)](proj_dir)
    reader.run()
    raw_concept_file = concept_detect(proj_dir)
    reduced_cpt_file = os.path.join(proj_dir, "reduced_concepts.jsonl")
    remove_regular_concpts(reg_cpt_file, raw_concept_file, reduced_cpt_file)


if __name__ == "__main__":
    """
    Read raw materials from projects and produce the tokens and recoginize the concepts from the artifacts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", default="../../data/projects/CCHIT", help="the directory of the project dataset"
    )
    parser.add_argument("--reg_cpt_file", default="../../data/regular_concepts.csv",
                        help="the csv file of regular concepts")
    args = parser.parse_args()
    preprocess(args.project_dir, args.reg_cpt_file)
