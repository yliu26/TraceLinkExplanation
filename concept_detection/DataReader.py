import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import json
import os
from pathlib import Path

SART_CSV, TART_CSV, LK_CSV = "source_artifacts.csv", "target_artifacts.csv", "links.csv"
cur_dir = str(Path(__file__).parent.absolute())


class TraceReader:
    def __init__(self, out_dir):
        self.sarts, self.tarts = dict(), dict()
        self.links = defaultdict(set)
        self.out_dir = out_dir

    def _read_art(self):
        raise NotImplementedError

    def _read_link(self):
        raise NotImplementedError

    def flat_links(self):
        flink = []
        for sid in self.links:
            for tid in self.links[sid]:
                flink.append((sid, tid))
        return flink

    def to_csv(self, sart_csv=SART_CSV, tart_csv=TART_CSV, link_csv=LK_CSV):
        df = pd.DataFrame()
        df["id"] = self.sarts.keys()
        df["arts"] = self.sarts.values()
        df.to_csv(os.path.join(self.out_dir, sart_csv), index=False)

        df = pd.DataFrame()
        df["id"] = self.tarts.keys()
        df["arts"] = self.tarts.values()
        df.to_csv(os.path.join(self.out_dir, tart_csv), index=False)

        df = pd.DataFrame()
        df["sid"] = [x[0] for x in self.flat_links()]
        df["tid"] = [x[1] for x in self.flat_links()]
        df.to_csv(os.path.join(self.out_dir, link_csv), index=False)

    def run(self):
        self._read_art()
        self._read_link()
        self.to_csv()
        return self


class CM1Reader(TraceReader):
    def __init__(
        self,
        data_dir=cur_dir + "/data/CM1",
        sart_file="CM1-sourceArtifacts.xml",
        tart_file="CM1-targetArtifacts.xml",
        link_file="CM1-answerSet.xml",
    ):
        super().__init__(out_dir=data_dir)
        self.sart_file, self.tart_file = (
            os.path.join(data_dir, sart_file),
            os.path.join(data_dir, tart_file),
        )
        self.link_file = os.path.join(data_dir, link_file)

    def _read_art(self):
        def read(f):
            res = dict()
            root = ET.parse(f).getroot()
            for art in root.iter("artifact"):
                id = art.find("id").text
                arts = art.find("content").text.strip("\n\t\r ").replace("\n", " ")
                res[id] = arts
            return res

        self.sarts, self.tarts = read(self.sart_file), read(self.tart_file)

    def _read_link(self):
        root = ET.parse(self.link_file).getroot()
        for lk in root.iter("link"):
            sid = lk.find("source_artifact_id").text
            tid = lk.find("target_artifact_id").text
            self.links[sid].add(tid)


class CCHITReader(TraceReader):
    """
    Convert CCHIT in XML into csv files
    """

    def __init__(
        self,
        data_dir=cur_dir + "/data/CCHIT",
        sart_file="source.xml",
        tart_file="target.xml",
        link_file="answer2.xml",
    ):
        super().__init__(out_dir=data_dir)
        self.sart_file, self.tart_file = (
            os.path.join(data_dir, sart_file),
            os.path.join(data_dir, tart_file),
        )
        self.link_file = os.path.join(data_dir, link_file)

    def _read_art(self):
        def read(f):
            res = dict()
            root = ET.parse(f).getroot()
            for art in root.iter("artifact"):
                id = art.find("art_id").text
                arts = art.find("art_title").text
                res[id] = arts
            return res

        self.sarts, self.tarts = read(self.sart_file), read(self.tart_file)

    def _read_link(self):
        root = ET.parse(self.link_file).getroot()
        for lk in root.iter("link"):
            sid = lk.find("source_artifact_id").text
            tid = lk.find("target_artifact_id").text
            self.links[sid].add(tid)


class DronologyReader(TraceReader):
    """
    Convert the dronology json file to csv
    """

    def __init__(self, data_dir=cur_dir + "/data/Dronology/"):
        super().__init__(out_dir=data_dir)
        js_file = os.path.join(data_dir, "dronologydataset01.json")
        with open(js_file, encoding="utf8") as fin:
            self.entries = json.load(fin)["entries"]

    def _read_art(self):
        for entry in self.entries:
            art_id = entry["issueid"]
            attributes = entry["attributes"]
            art_summary = attributes["summary"].strip("\n\t\r")
            art_describ = attributes["description"].strip("\n\t\r")
            art_arts = art_summary + art_describ
            self.arts[art_id] = art_arts

    def _read_link(self):
        for entry in self.entries:
            sid = entry["issueid"]
            children = entry["children"]
            for child in children:
                for tid in children[child]:
                    self.links[sid].add(tid)


class PTCReader(TraceReader):
    def __init__(
        self,
        data_dir=cur_dir + "/data/PTC/",
        sart_file="SDD2.xml",
        tart_file="SRS.xml",
        link_file="SDD2SRS.txt",
    ):
        super().__init__(out_dir=data_dir)
        self.data_dir = data_dir
        self.sart_file, self.tart_file = (
            os.path.join(data_dir, sart_file),
            os.path.join(data_dir, tart_file),
        )
        self.link_file = os.path.join(data_dir, link_file)

    def _read_art(self):
        def read(f):
            arts = dict()
            root = ET.parse(f).getroot()
            for art in root.iter("artifact"):
                art_id = art.find("art_id").text
                art_title = art.find("art_title").text
                if not art_id or not art_title:
                    continue
                art_title = art_title.strip('"\n\r\t\s ')
                arts[art_id] = art_title
            return arts

        self.sarts = read(self.sart_file)
        self.tarts = read(self.tart_file)

    def _read_link(self):
        df = pd.read_csv(self.link_file)
        sids = df.iloc[:, 0]
        tids = df.iloc[:, 1]
        for sid, tid in zip(sids, tids):
            self.links[sid].add(tid)


class InfusionPumpReader(TraceReader):
    def __init__(
        self,
        data_dir=cur_dir + "/data/PTC/",
    ):
        pass


if __name__ == "__main__":
    # readers = [CCHITReader(), DronologyReader(), PTCReader()]
    readers = [CM1Reader()]
    for x in readers:
        x.run()
