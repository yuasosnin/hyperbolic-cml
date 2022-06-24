from typing import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geoopt


class CML(nn.Module):
    '''
    https://github.com/hand10ryo/PyTorchCML/blob/main/PyTorchCML/models/CollaborativeMetricLearning.py
    '''
    def __init__(
        self, 
        n_user: int, 
        n_item: int, 
        embedding_dim: int = 16, 
        dropout_rate: float = 0.5, 
        **kwargs
    ) -> None:
        super().__init__()
        
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(self.n_user, self.embedding_dim, **kwargs)
        self.item_embedding = nn.Embedding(self.n_item, self.embedding_dim, **kwargs)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        torch.nn.init.uniform_(self.user_embedding.weight, a=-0.2, b=0.2)
        torch.nn.init.uniform_(self.item_embedding.weight, a=-0.2, b=0.2)
    
    def from_pretrained(self, *args, **kwargs) -> None:
        raise NotImplemented

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> Tuple[torch.Tensor]:
        user_embedding = self.user_embedding(users)
        pos_item_embedding = self.dropout(self.item_embedding(pos_items))
        neg_item_embedding = self.dropout(self.item_embedding(neg_items))

        return user_embedding, pos_item_embedding, neg_item_embedding

    def predict(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> Tuple[torch.Tensor]:
        user_embedding, pos_item_embedding, neg_item_embedding = self.forward(users, pos_items, neg_items)
        pos_item_dist = torch.cdist(user_embedding, pos_item_embedding)[:, 0, :]
        neg_item_dist = torch.cdist(user_embedding, neg_item_embedding)[:, 0, :]

        return pos_item_dist, neg_item_dist
    

class HyperEmbedding(nn.Module):
    '''
    https://github.com/mil-tokyo/hyperbolic_nn_plusplus/blob/main/geoopt_plusplus/modules/embedding.py
    '''
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None, 
        norm_type: float = 2.0, 
        scale_grad_by_freq: bool = False,
        sparse: bool = False, 
        _weight: Optional[torch.Tensor] = None, 
        init_std: float = 1e-2, 
        manifold: Optional[geoopt.Manifold] = None
    ) -> None:
        super().__init__()
        
        if manifold is None:
            self.manifold = geoopt.Euclidean()
        else:
            self.manifold = manifold
        self.init_std = init_std
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = geoopt.ManifoldParameter(torch.empty(num_embeddings, embedding_dim), manifold=manifold)
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
        self.sparse = sparse

    def reset_parameters(self) -> None:
        with torch.no_grad():
            data = self.manifold.random_normal(
                *self.weight.shape, mean=0, std=1e-2
            ).data
            self.weight.data.copy_(data)
            if self.padding_idx is not None:
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse))
    
    
class HyperCML(nn.Module):
    def __init__(
        self, 
        n_user: int, 
        n_item: int, 
        embedding_dim: int = 16, 
        dropout_rate: float = 0.5, 
        manifold: Optional[geoopt.Manifold] = None, 
        **kwargs
    ) -> None:
        super().__init__()
        
        if manifold is None:
            self.manifold = geoopt.PoincareBall()
        else:
            self.manifold = manifold
        
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim

        self.user_embedding = HyperEmbedding(self.n_user, self.embedding_dim, manifold=self.manifold, **kwargs)
        self.item_embedding = HyperEmbedding(self.n_item, self.embedding_dim, manifold=self.manifold, **kwargs)
        self.dropuot = nn.Dropout(p=dropout_rate)

    def from_pretrained(self, *args, **kwargs) -> None:
        raise NotImplemented

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> Tuple[torch.Tensor]:
        user_embedding = self.user_embedding(users)
        pos_item_embedding = self.dropuot(self.item_embedding(pos_items))
        neg_item_embedding = self.dropuot(self.item_embedding(neg_items))

        return user_embedding, pos_item_embedding, neg_item_embedding

    def predict(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> Tuple[torch.Tensor]:
        user_embedding, pos_item_embedding, neg_item_embedding = self.forward(users, pos_items, neg_items)
        pos_item_dist = self.manifold.dist(user_embedding, pos_item_embedding)
        neg_item_dist = self.manifold.dist(user_embedding, neg_item_embedding)

        return pos_item_dist, neg_item_dist
