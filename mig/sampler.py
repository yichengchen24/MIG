import torch

from enum import Enum
from abc import ABCMeta, abstractmethod
from typing import Iterator, Optional

from .data import DataPoint, Dataset
from .label import LabelGraph

Tensor = torch.Tensor


class SamplerType(Enum):
    MIG = "mig"
    RANDOM = "random"
    IFD = "ifd"
    

class Sampler(metaclass=ABCMeta):
    """Abstract base class for samplers."""
    
    @abstractmethod
    def sample(self, pool: Dataset, num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        """Sample a subset of the pool with 'num_sample' samples."""


class RandomSampler(Sampler):
    
    def sample(self, pool: Dataset[DataPoint], num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        if num_sample >= len(pool):
            raise ValueError("num_sample must be less than the size of the pool.")
        import random
        rng = random.Random(42)
        indices = rng.sample(range(len(pool)), num_sample)
        for i in indices:
            yield pool[i]


class IFDSampler(Sampler):
    """
    Sample data by IFD
    """
    
    def __init__(
        self,
        upper_bound: float = 1.0,
        lower_bound: float = 0.0,
    ):
        super().__init__()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    
    def filter(self, pool: Dataset[DataPoint]):
        """Filter outilers"""
        
        filtered_pool = []
        for dp in pool:
            if dp.score <= self.upper_bound and dp.score >= self.lower_bound:
                filtered_pool.append(dp)
        
        return filtered_pool
        
    def sample(self, pool: Dataset[DataPoint], num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        
        # filter
        filtered_pool = self.filter(pool)
        
        # sort
        sorted_pool = sorted(filtered_pool, key=lambda x: x.score, reverse=True)
        
        # sample
        for i in range(num_sample):
            yield sorted_pool[i]
    
    
class MIGSampler(Sampler):
    """Maximum Information Gain Sampler.
    
    Args:
        label_graph
    """
    
    def __init__(
        self,
        label_graph: LabelGraph,
        phi_type: str = "pow",
        phi_alpha: float = 1.0,
        phi_a: float = 1e-6,
        phi_b: float = 0.8,
        prop_weight: float = 1.0,
        norm: bool = True
    ):
        super().__init__()
        self.label_graph = label_graph
        self.phi_type = phi_type
        self.phi_alpha = phi_alpha
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.prop_weight = prop_weight
        self.norm = norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def d_phi(self, x: Tensor) -> Tensor:
        """Compute the gradient of the total information.
        
        Args:
            x (Tensor): Information distribution in 1D.
        Returns:
            Tensor: Gradient of the total information.
        """
        
        if self.phi_type == "exp":
            return self._d_phi_exp(x)
        elif self.phi_type == "pow":
            return self._d_phi_pow(x)
        else:
            raise ValueError(f"Invalid phi_type: {self.phi_type}")
        
    def _d_phi_exp(self, x: Tensor) -> Tensor:
        return self.phi_alpha * torch.exp(-self.phi_alpha * x)
    
    def _d_phi_pow(self, x: Tensor) -> Tensor:
        return self.phi_b * (x + self.phi_a) ** (self.phi_b - 1)
        
    def calc_prop_matrix(self):
        # get the weighted adjacency matrix
        mat_w = self.label_graph.wam # (n_labels, n_labels)
        if mat_w is None:
            raise ValueError("Weighted adjacency matrix is not set.")
        
        # get the propagation matrix
        n = mat_w.size(0)
        mask = torch.eye(n, device=mat_w.device)
        mat_p_diag = mat_w * mask
        mat_p_ndiag = mat_w * (1 - mask) * self.prop_weight
        mat_p = mat_p_diag + mat_p_ndiag
        
        # normalization
        mat_p = mat_p / mat_p.sum(axis=0)
        
        mat_p = mat_p.to(self.device).float()
        
        return mat_p
        
    def sample(self, pool: Dataset, num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        """Sample a subset of the pool with 'num_sample' samples."""
        # make a copy of the pool
        pool = pool.copy()
        
        # get the propagation matrix
        mat_p = self.calc_prop_matrix()
        
        # get vectorized pool, which also represent the information distribution
        vec_pool = self.label_graph.vectorize(pool, self.norm).to(self.device).float() # (n_samples, n_labels)
        vec_pool_prop = vec_pool @ mat_p
        
        # initialize the selected samples and the information gain (score) distribution
        n_sel = 0
        mask = torch.ones(len(pool), dtype=torch.bool, device=self.device)
        vec_x_sel = torch.zeros(len(self.label_graph), device=self.device, dtype=torch.float32)
        
        while n_sel < num_sample:
            # gradient of the total information
            vec_g = self.d_phi(vec_x_sel).to(self.device).float() # (n_labels,)
            
            # calculate information gain of each data point
            if not batch_size:
                vec_candidate = vec_pool_prop[mask] @ vec_g # (n_candidates,)
            else:
                vec_candidate_list = []
                total_size = vec_pool_prop.shape[0]
                for i in range(0, total_size, batch_size):
                    # get indices for current batch
                    end_index = min(i + batch_size, total_size)
                    # process batch data
                    batch_vec_pool_prop = vec_pool_prop[i:end_index]
                    batch_mask = mask[i:end_index]
                    batch_masked_vec_pool_prop = batch_vec_pool_prop[batch_mask]
                    # check len
                    if batch_masked_vec_pool_prop.shape[0] == 0:
                        continue
                    # calculate information gain of each data point
                    vec_candidate_batch = batch_masked_vec_pool_prop @ vec_g
                    vec_candidate_list.append(vec_candidate_batch)
                # merge results
                vec_candidate = torch.cat(vec_candidate_list, dim=0)
            
            # select the data point with the maximum information gain
            idx = torch.argmax(vec_candidate)
            indices = torch.nonzero(mask, as_tuple=False).squeeze()
            selected_idx = indices[idx]
            
            vec_x_sel += vec_pool_prop[selected_idx]
            mask[selected_idx] = False
            n_sel += 1
            dp = pool.pop(idx.item())
            
            yield dp


def create_sampler(sampler_type: SamplerType, label_graph: Optional[LabelGraph], phi_type: str, phi_alpha: float, phi_a: float, phi_b: float, prop_weight: float, norm: bool) -> "Sampler":
    if sampler_type == SamplerType.MIG:
        if label_graph is None:
            raise ValueError("label_graph must be provided for MIG sampler.")
        return MIGSampler(label_graph, phi_type=phi_type, phi_alpha=phi_alpha, phi_a=phi_a, phi_b=phi_b, prop_weight=prop_weight, norm=norm)
    elif sampler_type == SamplerType.RANDOM:
        return RandomSampler()
    