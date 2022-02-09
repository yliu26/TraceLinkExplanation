import os
from asyncio import start_server

import pandas as pd
from graph_tools import Graph
from pathlib import Path
import sys
from multiprocessing import Pool
from graphviz import Source
from jsonlines import jsonlines
from stanza.server import CoreNLPClient
from nltk.stem import WordNetLemmatizer
from pattmatch import kmp

sys.path.append("..")

EDGE_VERB = "verb"


class RelationGraph:
    def __init__(self, bidirection=True):
        """
        Data structure for storing the relationships between the concepts
        """
        self.lemm = WordNetLemmatizer()
        self.g = Graph()
        self.bidirection = bidirection

    def process_concept_name(self, concept):
        tokens = concept.lower().split()
        tokens = [self.lemm.lemmatize(x) for x in tokens]
        return " ".join(tokens)

    def add_vertex(self, concept):
        """
        Add vertex if not exist
        """
        self.g.add_vertex(self.process_concept_name(concept))

    def add_relation(self, relation):
        self._add_relation(relation)
        if self.bidirection:
            self._add_relation(reversed(relation))

    def find_vague_match(self, concept):
        c_tokens = concept.split()
        for v in self.g.vertices():
            v_tokens = v.split()
            if len(kmp(c_tokens, v_tokens)) or len(kmp(v_tokens, c_tokens)):
                return v
        return None

    def is_reachable(self, c1, c2, vague=False):
        c1, c2 = self.process_concept_name(c1), self.process_concept_name(c2)
        c1_match = c1 if c1 in self.g.vertices() else self.find_vague_match(c1)
        c2_match = c2 if c2 in self.g.vertices() else self.find_vague_match(c2)
        if c1_match == None or c2_match == None:
            return False
        elif c1_match == c2_match:
            return True
        return self.g.is_reachable(c1_match, c2_match) is not None

    def find_path(self, c1, c2):
        c1, c2 = self.process_concept_name(c1), self.process_concept_name(c2)
        c1_match = c1 if c1 in self.g.vertices() else self.find_vague_match(c1)
        c2_match = c2 if c2 in self.g.vertices() else self.find_vague_match(c2)
        if c1_match == None or c2_match == None:
            return False
        all_path = self.g.shortest_paths(c1_match, c2_match)
        if len(all_path) == 0:
            return []
        path = all_path[0]
        if len(path) == 1:
            return (c1, "synonym", c2)
        elif len(path) > 1:
            path_tup = []
            for cur in range(1, len(path)):
                pnode = path[cur - 1]
                cnode = path[cur]
                verb = list(
                    self.g.get_edge_attribute_by_id(pnode, cnode, 0, EDGE_VERB)
                )[0]
                path_tup.append((pnode, verb, cnode))
            if c1 not in self.g.vertices():
                path_tup.insert(0, (c1, "match", path_tup[0][0]))
            if c2 not in self.g.vertices():
                path_tup.append((path_tup[-1][-1], "match", c2))
            return path_tup
        else:
            return []

    def _add_relation(self, relation):
        u, verb, v = relation
        u = self.process_concept_name(u)
        v = self.process_concept_name(v)
        if not self.g.has_edge(u, v):
            self.g.add_edge(u, v)
            verbs = {} if verb is None else {verb}
        else:
            verbs = self.g.get_edge_attribute_by_id(u, v, 0, EDGE_VERB)
            if verb is not None:
                verbs.add(verb)
        self.g.set_edge_attribute_by_id(u, v, 0, EDGE_VERB, verbs)

    @property
    def concepts(self):
        return self.g.vertices()

    @property
    def links(self):
        return self.g.unique_edges()

    def merge(self, graph):
        for v in graph.g.vertices():
            self.g.add_vertex(v)
        for (u, v) in graph.g.edges():
            self.g.add_edge(u, v)
            self.g.add_edge(v, u)
            verbs = self.g.get_edge_attribute_by_id(u, v, 0, EDGE_VERB)
            if not verbs:
                verbs = set()
            verbs = verbs.update(graph.g.get_edge_attribute_by_id(u, v, 0, EDGE_VERB))
            self.g.set_edge_attribute_by_id(u, v, 0, EDGE_VERB, verbs)

    def clean_graph(self, anchors, cores=4):
        """
        Clean the graph by removing the concepts (and corresponding edges) if they do not link to an anchor-point concpet.
        """
        keep = set()
        with Pool(cores) as pool:
            res = pool.starmap(
                self.clean_worker, [(i, anchors, self.g) for i, a in enumerate(anchors)]
            )
            for r in res:
                keep.update(r)

        for v in self.g.vertices():
            if v not in keep:
                self.g.delete_vertex(v)

    @staticmethod
    def clean_worker(idx, anchors, graph):
        keep = set()
        s = anchors[idx]
        for i in range(idx, len(anchors)):
            t = anchors[i]
            for p in graph.shortest_paths(s, t):
                for n in p:
                    keep.add(n)
        return keep

    def _load(self, concepts, links, verbs):
        for c in concepts:
            self.g.add_vertex(c)
        for l in links:
            for r in links[l]:
                self.g.add_edge(l, r)
                self.g.add_edge(r, l)
                self.g.set_edge_attribute_by_id(l, r, 0, EDGE_VERB, verbs[l, r])
                self.g.set_edge_attribute_by_id(r, l, 0, EDGE_VERB, verbs[l, r])
        return self

    def load(self, link_file):
        if link_file.endswith("csv"):
            rdf = pd.read_csv(link_file)
            for idx, row in rdf.iterrows():
                lc, rc = row["left"].lower(), row["right"].lower()
                self.add_relation((lc, None, rc))
                verbs = set(eval(row["verbs"]))
                self.g.set_edge_attribute_by_id(lc, rc, 0, EDGE_VERB, verbs)
        elif link_file.endswith("jsonl"):
            with jsonlines.open(link_file) as fin:
                for obj in fin:
                    lc, rc = obj["left"].lower(), obj["right"].lower()
                    verb = obj["verb"]
                    self.add_relation((lc, None, rc))
                    self.g.set_edge_attribute_by_id(lc, rc, 0, EDGE_VERB, verb)

        return self

    def dump(self, dir, link_file):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        rfile = os.path.join(dir, link_file)
        rdf = pd.DataFrame(columns=["left", "right", "verbs"])
        for lc, rc in self.g.edges():
            rdf = rdf.append(
                {
                    "left": lc,
                    "right": rc,
                    "verbs": self.g.get_edge_attribute_by_id(lc, rc, 0, EDGE_VERB),
                },
                ignore_index=True,
            )
        rdf.to_csv(rfile, index=False)

    def draw(self, filename):
        s = self.g.export_dot()
        Source(s).render(filename)

    def __contains__(self, concept):
        concept = self.process_concept_name(concept)
        return self.g.has_vertex(concept)


if __name__ == "__main__":
    rel_g = RelationGraph()
    rel_g.add_vertex("Apple")
    rel_g.add_vertex("apples")
    rel_g.add_vertex("orange")
    rel_g.add_vertex("fruit")
    rel_g.add_relation(("orange", "is", "fruit"))
    rel_g.add_relation(("apple", "is", "fruit"))
    print(rel_g.g.unique_edges())
    print(rel_g.is_reachable("Apple", "orange"))
