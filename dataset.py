import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geoopt

# TODO: rewrite sampling in torch and therefore on cuda
class PariwiseDataset(Dataset):
    def __init__(self, data, n_users, n_items, neg_samples=1, weights=None, seed=None):
        self.data = data
        self.n_users = n_users
        self.n_items = n_items
        self.neg_samples = neg_samples
        self.weights = weights
        self.random_gen = np.random.default_rng(seed=seed)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        user = self.data.loc[idx, 'user_id']
        item_pos = self.data.loc[idx, 'item_id']

        # negative mining
        rated_items = self.data.loc[self.data['user_id'] == user, 'item_id']
        possible_neg = np.setdiff1d(np.arange(self.n_items), rated_items)
        if self.weights is not None:
            weights = self.weights[possible_neg] / self.weights[possible_neg].sum()
        else:
            weights = None
        item_neg = self.random_gen.choice(
            possible_neg, 
            size=self.neg_samples, 
            replace=True, 
            p=weights)
        
        sample = [
            torch.tensor([user]), 
            torch.tensor([item_pos]), 
            torch.tensor(item_neg)
        ]
        return sample