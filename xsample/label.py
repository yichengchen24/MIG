import torch
import pickle
import pandas as pd

from enum import Enum
from abc import ABCMeta, abstractmethod
from typing import Optional, Union
from sentence_transformers import SentenceTransformer

from .data import DataPoint


Tensor = torch.Tensor


class LabelGraphType(Enum):
    SIM = "sim"
    

class LabelGraph(metaclass=ABCMeta):
    
    def __init__(self):
        self._labels: list[str] = []
        self._label_to_index: dict[str, int] = {}
        self._W: None | Tensor = None
    
    def load(self, filename: str):
        """Load the label graph from a file."""
        items = ["_labels", "_label_to_index", "_W"]
        with open(filename, "rb") as f:
            data = pickle.load(f)
            for item in items:
                setattr(self, item, data[item])
    
    def dump(self, filename: str):
        """Dump the label graph to a file."""
        items = ["_labels", "_label_to_index", "_W"]
        data = {item: getattr(self, item) for item in items}
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    
    def index_label(self, label: str) -> int:
        """Return the index of a label."""
        return self._label_to_index[label]
    
    def _build_from_dataset(self, dataset: list[DataPoint]):
        """Build the label graph from the dataset."""
        self._labels = sorted(set([label for dp in dataset for label in dp.labels]))
        self._label_to_index = {label: i for i, label in enumerate(self._labels)}
        self._W = self.build_wam(self._labels)
    
    @property
    def labels(self) -> list[str]:
        """Return the list of label names."""
        return self._labels.copy()
    
    @property
    def wam(self) -> Tensor:
        """Return the weighted adjacency matrix of the label set."""
        if self._W is None:
            raise ValueError("Weighted adjacency matrix is not set.")
        return self._W
    
    def build_wam(self, label_list: list[str] | list[int]) -> Tensor:
        """Build the weighted adjacency matrix of the label set."""
        try:
            return self.calc_weighted_adjacency_matrix(label_list)
        except NotImplementedError:
            # fall back to calculate edge weight for each pair of labels
            wam = torch.zeros((len(label_list), len(label_list)))
            for i, label1 in enumerate(label_list):
                for j, label2 in enumerate(label_list[:i]):
                    wam[i, j] = wam[j, i] = self.calc_weight(label1, label2)
    
    def visualize(self, backend: str = "jaal"):
        """Visualize the label graph."""
        if backend == "jaal":
            try:
                from jaal import Jaal
            except:
                raise ImportError("Please install jaal to use the Jaal backend.")
            
            # construct
            edges = pd.DataFrame([{"from": i, "to": j, "weight": 100 * self.wam[i, j]} for i in range(len(self._labels)) for j in range(i) if self.wam[i, j] > 0])
            nodes = pd.DataFrame({"id": range(len(self._labels)), "title": self._labels})
            
            # visualize
            port = 8050
            while True:
                try:
                    Jaal(edges, nodes).plot(port=port)
                    break
                except OSError as e:
                    if "Address already in use" in str(e):
                        port += 1
                    else:
                        raise
    
    @abstractmethod
    def calc_weighted_adjacency_matrix(self, label_list: list[str] | list[int]) -> Tensor:
        """Calculate the weighted adjacency matrix of the label set."""
        
    @abstractmethod
    def calc_weight(self, label1: Union[str, int], label2: Union[str, int]) -> float:
        """Calculate the weight between two labels."""
        
    def vectorize(self, data: Union[DataPoint, list[DataPoint]], norm: bool = False) -> Tensor:
        """Vectorize the data points. Each row is a vetorized data point."""
        
        if isinstance(data, DataPoint):
            return self.vectorize([data])[0]
        
        vec = torch.zeros((len(data), len(self._labels)))
        for i, dp in enumerate(data):
            n_labels = len(dp.labels)
            for label in dp.labels:
                j = self.index_label(label)
                vec[i, j] = dp.score if not norm else dp.score / n_labels
        
        return vec
            
    def __len__(self):
        return len(self._labels)
    

class SimLabelGraph(LabelGraph):
    
    def __init__(
        self,
        dataset: Optional[list[DataPoint]] = None,
        *,
        load_from: Optional[str] = None,
        embedding_model: str = "",
        embedding_cache: str = "cache",
        sim_threshold: float = 0.9,
    ):
        super().__init__()
        self._sim_threshold = sim_threshold
        self._embedding_model = SentenceTransformer(embedding_model, cache_folder=embedding_cache)
        
        if load_from is not None:
            self.load(load_from)
        else:
            assert dataset is not None
            self._build_from_dataset(dataset)
    
    def calc_weighted_adjacency_matrix(self, label_list: list[str] | list[int]) -> torch.Tensor:
        """Calculate the weighted adjacency matrix of the label set."""
        
        label_list = [label if isinstance(label, str) else self._labels[label] for label in label_list]
        
        embeddings = self._embedding_model.encode(label_list, convert_to_tensor=True, normalize_embeddings=True) # (N, D)
        
        # calculate cosine similarity for normalize embedding
        wam = embeddings @ embeddings.T # (N, N)
        
        # set similarity below threshold to zero
        wam = wam.where(wam >= self._sim_threshold, 0.0)
        
        return wam
    
    def calc_weight(self, label1: str | int, label2: str | int) -> float:
        
        labels = [label if isinstance(label, str) else self._labels[label] for label in (label1, label2)]
        
        embeddings = self._embedding_model.encode(labels, convert_to_tensor=True, normalize_embeddings=True) # (2, D)
        
        sim = torch.dot(embeddings[0], embeddings[1]).item()
        
        if sim < self._sim_threshold:
            sim = 0.0
        
        return sim


def create_label_graph(graph_type: LabelGraphType, dataset: Optional[list[DataPoint]] = None, *, load_from: Optional[str] = None, embedding_model: Optional[str] = "", sim_threshold: Optional[float] = 0.9) -> "LabelGraph":
    
    if graph_type == LabelGraphType.SIM:
        return SimLabelGraph(dataset=dataset, load_from=load_from, embedding_model=embedding_model, sim_threshold=sim_threshold)
    else:
        raise NotImplementedError
    