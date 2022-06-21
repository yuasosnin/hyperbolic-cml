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


class AdditiveLoss(nn.Module):
    '''
    A simple wrapper for multiple loss functions, adding together
    '''
    
    def __init__(self, *args, coefficients: Optional[Iterable] = None) -> None:
        super().__init__()
        self.losses = args
        if coefficients is not None:
            assert len(args) == len(coefficients)
            self.coefficients = coefficients
        else:
            self.coefficients = [1]*len(args)

    def forward(self, *args) -> torch.Tensor:
        loss = 0
        for criterion, coefficient in zip(self.losses, self.coefficients):
            loss += coefficient * criterion(*args)
        return loss
    
    
def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    batch_size, items_size, embedding_dim = x.shape
    losses = torch.zeros(items_size)
    for i in range(items_size):
        c = torch.cov(x[:, i, :].transpose(0,1), correction=0)
        c_norm = torch.linalg.norm(c, ord='fro')
        c_diag_norm = torch.linalg.norm(torch.diag(c))
        losses[i] = (1/batch_size) * (c_norm - c_diag_norm**2)
    return losses.mean()


class CovarianceLoss(nn.Module):
    '''
    CML loss from ...
    '''
    
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor
        ) -> torch.Tensor:
        
        loss = 0
        for x in (anchor, positive, negative):
            loss += covariance_loss(x)
        return loss
    

class DistortionLoss(nn.Module):
    '''
    CML loss from ...
    '''
    
    def __init__(self, manifold: geoopt.Manifold, reduction: Literal['mean', 'sum'] = 'mean', eps: float = 1e-6) -> None:
        super().__init__()
        self.manifold = manifold
        self.reduction = reduction
        self.eps = eps

    def _distortion(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        hyper_dist = self.manifold.dist(u, v)#**0.5
        euclid_dist = torch.cdist(self.manifold.logmap0(u), self.manifold.logmap0(v))[:, 0, :] #**0.5
        return torch.abs(hyper_dist - euclid_dist) / (euclid_dist + self.eps)
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor
        ) -> torch.Tensor:
        
        pos_distortion = self._distortion(anchor, positive)
        neg_distortion = self._distortion(anchor, negative)

        if self.reduction == 'mean':
            return torch.mean(pos_distortion) + torch.mean(neg_distortion)
        elif self.reduction == 'sum':
            return torch.sum(pos_distortion) + torch.sum(neg_distortion)
        elif self.reduction == 'none':
            return pos_distortion + neg_distortion
