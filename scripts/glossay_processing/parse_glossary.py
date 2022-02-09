import pandas as pd
import os
import jsonlines


def is_acronym(term):
    for c in term:
        if c.isalpha() and not c.isupper():
            return False
    return True


def write_glossary(out_dir, acrn_dict, def_dict):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    acrn_file = os.path.join(out_dir, "glossary_acronym.jsonl")
    def_file = os.path.join(out_dir, "glossary_definition.jsonl")

    with jsonlines.open(acrn_file, "w") as fout:
        for a in acrn_dict:
            fout.write({"short": a, "long": [acrn_dict[a]]})

    with jsonlines.open(def_file, "w") as fout:
        for d in def_dict:
            fout.write({"concept": d, "definition": [def_dict[d]]})


def parse_csv_glossary(path):
    df = pd.read_csv(path)
    acrn_dict, def_dict = dict(), dict()
    df = df.fillna("")
    for idx, row in df.iterrows():
        acrn, concept, definition = row["Acronym"], row["Concept"], row["Definition"]

        if len(acrn) == 0 and len(concept) == 0:
            continue

        if len(acrn) > 0 and len(concept) > 0:
            acrn_dict[acrn] = concept

        if len(definition) > 0:
            if len(concept) > 0:
                def_dict[concept] = definition
            elif len(acrn) > 0:
                def_dict[acrn] = definition
    return acrn_dict, def_dict


def parse_cchit():
    # split cchit glossary into acronym, concept, and definition
    df = pd.read_csv("cchit_raw.csv")
    acrn_dict, def_dict = dict(), dict()
    for i, row in df.iterrows():
        term, par = row[0], row[1]
        if term != term or par != par:
            continue
        sents = par.split("\n")
        s1 = sents[0]
        rest = "\n".join(sents[1:])
        if is_acronym(term) and not s1.endswith("."):
            acronym = term
            concept = s1
            definition = rest
        else:
            concept = term
            definition = par

        acrn_dict[acronym] = concept
        if len(definition) > 0:
            def_dict[concept] = definition

    return acrn_dict, def_dict


if __name__ == "__main__":
    acrn_dict, def_dict = parse_cchit()
    write_glossary("./CCHIT", acrn_dict, def_dict)
    acrn_dict, def_dict = parse_csv_glossary("./cm1_raw.csv")
    write_glossary("./CM1", acrn_dict, def_dict)
    acrn_dict, def_dict = parse_csv_glossary("./ptc_raw.csv")
    write_glossary("./PTC", acrn_dict, def_dict)
