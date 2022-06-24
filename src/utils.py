from typing import *

import os, requests, zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geoopt


def load_movielens(version: Literal['100k', '1m'] = '100k', path: os.PathLike = 'data') -> pd.DataFrame:
    '''Download MovieLens data and get an interactions DataFrame of specified version'''
    
    link = f'https://files.grouplens.org/datasets/movielens/ml-{version}.zip'
    if not os.path.exists(f'{path}/ml-{version}'):
        with open('data.zip', 'wb') as f:
            f.write(requests.get(link, stream=True).content)
        zipfile.ZipFile('data.zip', 'r').extractall(path)
        os.remove('data.zip')

    if version == '1m':
        data = pd.read_csv(
            f'{path}/ml-{version}/ratings.dat', 
            delimiter='::', 
            header=None, 
            names=['user_id', 'item_id', 'rating', 'timestamp'], 
            engine='python')
        
    elif version == '100k':
        data = pd.read_csv(
            f'{path}/ml-{version}/u.data', 
            sep='\t', 
            header=None, 
            names=['user_id', 'item_id', 'rating', 'timestamp'])
        
    else:
        raise ValueError('Invalid version')

    data['user_id'] = np.unique(data['user_id'], return_inverse=True)[1]
    data['item_id'] = np.unique(data['item_id'], return_inverse=True)[1]
    data = data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    return data


def train_test_split_interations(
    data: pd.DataFrame, 
    method: Literal['last', 'random'] = 'last', 
    random_state: int = 1
) -> Tuple[pd.DataFrame]:
    '''
    Split implicit recommendation DataFrame of format data[['user_id', 'item_id']] 
    into train, validation and test DataFrames, with only one interaciton per user
    in validation and test DataFrames and everything else in train DataFrame.
    Parameters
        method: method of choosing validation and test sets.
        If 'last', chooses 2 last interactions, assuming the data is sorted.
        If 'random', chooses 2 random interacitons.
    '''
    
    cols = ['user_id', 'item_id']
    
    if method == 'last':
        data_test = data.groupby('user_id').nth(-1).reset_index() # last item as test set
        data_valid = data.groupby('user_id').nth(-2).reset_index() # 2nd last item as valid set
        data_train = pd.concat([data, data_test, data_valid]).drop_duplicates(keep=False).reset_index(drop=True)
        return data_train[cols], data_valid[cols], data_test[cols]
    
    elif method == 'random':
        data_test = data.groupby('user_id').sample(random_state=random_state).reset_index()[cols]
        data_train = pd.concat([data, data_test]).drop_duplicates(keep=False).reset_index()[cols]
        data_valid = data_train.groupby('user_id').sample(random_state=random_state).reset_index()[cols]
        data_train = pd.concat([data_train, data_valid]).drop_duplicates(keep=False).reset_index()[cols]
        return data_train[cols], data_valid[cols], data_test[cols]


def select_by_index(input: torch.Tensor, index) -> torch.Tensor:
    '''?'''
    out = torch.zeros_like(index)
    for i in range(input.shape[0]):
        out[i] = input[i, index[i]]
    return out

def isin_batch(elements, test_elements):
    '''?'''
    out = torch.zeros_like(elements)
    for i in range(elements.shape[0]):
        out[i] = torch.isin(elements[i], test_elements[i]).item()
    return out.bool()

def hit_rate(pos_items, items, item_dist, k=10):
    topk_dist = item_dist.topk(k=k, dim=1, largest=False)
    topk = select_by_index(items, topk_dist.indices)
    return isin_batch(pos_items, topk).sum().item(), pos_items.shape[0]

def ndcg(pos_items, items, item_dist, k=10):
    topk_dist = item_dist.topk(k=k, dim=1, largest=False)
    topk = select_by_index(items, topk_dist.indices)
    _, indices = (topk == pos_items).nonzero(as_tuple=True)
    dcg = (1 / torch.log2(indices+2))
    #idcg = (1/torch.log2(torch.arange(0,k)+2)).sum()
    return dcg.sum().item(), pos_items.shape[0]

def full_hit_rate(trainer, valid_set, data_train):
    lists = trainer.predict_lists().numpy().argsort()
    hits = 0
    for i in range(trainer.model.n_user):
        user, item, _ = valid_set[i]
        user, item = user.item(), item.item()
        positives = data_train.loc[data_train.user_id == user].item_id.to_numpy()
        candidates = lists[i][~np.isin(lists[i], positives)]
        hits += (item in candidates[:10])
    return hits / trainer.model.n_user
