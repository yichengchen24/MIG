import json
import jsonlines as jsonl

from pydantic import BaseModel
from pathlib import Path
from rich.progress import track
from typing import Iterator, List, ClassVar

from .utils import len_jsonls

class DataPoint(BaseModel):
    labels: list[str]
    score: float
    raw: dict
    
    @classmethod
    def from_dict(cls, d: dict):
        
        labels = d['labels']
        score = d['score']
        return cls(labels=labels, score=score, raw=d)


class MIGDataPoint(DataPoint):
    valid_tags: ClassVar[List[str]] = None
        
    @classmethod
    def init(
        cls,
        valid_tag_path: str = '',
    ):
        if cls.valid_tags is None:
            with open(valid_tag_path, 'r') as f:
                cls.valid_tags = json.load(f)
    
    @classmethod
    def from_dict(cls, d: dict):
        
        if cls.valid_tags is None:
            raise ValueError("Please init first.")
        
        # check keys
        if "annotation" not in d or not d["annotation"] or "instag" not in d["annotation"] or not d["annotation"]["instag"] or "deita" not in d["annotation"] or not d["annotation"]["deita"]:
            return None
        
        # score
        score = 0.0
        for i in range(len(d["annotation"]["deita"]["quality_scores"])):
            score += d["annotation"]["deita"]["quality_scores"][i] * d["annotation"]["deita"]["complexity_scores"][i]
        score = score / len(d["annotation"]["deita"]["quality_scores"])
        
        # labels
        labels = [t for t in d["annotation"]["instag"]["content"] if t in cls.valid_tags]
        
        return cls(labels=labels, score=score, raw=d)
        

Dataset = list[DataPoint]


def load_dataset(
    filename: str | Path,
    valid_tag_path: str = '',
) -> list[DataPoint]:
    """Load a dataset from a JSON Lines file."""
    data_points = []
    MIGDataPoint.init(valid_tag_path=valid_tag_path)
    with jsonl.open(filename) as reader:
        for d in track(reader, total=len_jsonls(filename)):
            data_point = MIGDataPoint.from_dict(d)
            if data_point:
                data_points.append(data_point)
    return data_points


def dump_dataset(dataset: Iterator[DataPoint], filename: str | Path):
    """Dump a dataset to a JSON Lines file."""
    with jsonl.open(filename, "w") as writer:
        for dp in track(dataset):
            writer.write(dp.raw)

 