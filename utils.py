import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geoopt


def load_data(version='ml-100k'):
    import os, requests, zipfile
    link = f'https://files.grouplens.org/datasets/movielens/{version}.zip'
    if not os.path.exists(version):
        with open('data.zip', 'wb') as f:
            f.write(requests.get(link, stream=True).content)
        zipfile.ZipFile('data.zip', 'r').extractall()
        os.remove('data.zip')

    if version == 'ml-1m':
        data = pd.read_csv(f'{version}/ratings.dat', delimiter='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        #items, users = None, None
    elif version == 'ml-100k':
        data = pd.read_csv(f'{version}/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
        # genres = pd.read_csv(f'{version}/u.genre', sep='|', header=None, encoding='latin-1').iloc[:, 0].to_list()
        # items = pd.read_csv(f'{version}/u.item', sep='|', header=None, encoding='latin-1')
        # items.columns = ['item_id', 'title', 'release_date', 'empty', 'link'] + genres
        # items = items.drop(['empty', 'release_date', 'link'], axis=1)
        # items['item_id'] = items['item_id'] - 1
        # users = pd.read_csv(f'{version}/u.user', sep='|', header=None, encoding='latin-1', names='user_id,age,gender,occupation,zip_code'.split(','))
        # users['user_id'] = users['user_id'] - 1

    data['user_id'] = np.unique(data['user_id'], return_inverse=True)[1]
    data['item_id'] = np.unique(data['item_id'], return_inverse=True)[1]
    data = data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    return data#, items, users

def train_test_split_single(data, how='last', seed=1):
    cols = ['user_id', 'item_id']
    if how == 'last':
        data_test = data.groupby('user_id').nth(-1).reset_index() # last item as test set
        data_valid = data.groupby('user_id').nth(-2).reset_index() # 2nd last item as valid set
        data_train = pd.concat([data, data_test, data_valid]).drop_duplicates(keep=False).reset_index(drop=True)
        return data_train[cols], data_valid[cols], data_test[cols]
    elif how == 'random':
        data_test = data.groupby('user_id').sample(random_state=seed).reset_index()[cols]
        data_train = pd.concat([data, data_test]).drop_duplicates(keep=False).reset_index()[cols]
        data_valid = data_train.groupby('user_id').sample(random_state=seed).reset_index()[cols]
        data_train = pd.concat([data_train, data_valid]).drop_duplicates(keep=False).reset_index()[cols]
        return data_train[cols], data_valid[cols], data_test[cols]

def select_by_index(input, index):
    out = torch.zeros_like(index)
    for i in range(input.shape[0]):
        out[i] = input[i, index[i]]
    return out

def isin_batch(elements, test_elements):
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
    dcg = (1/torch.log2(indices+2))
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
    return hits/trainer.model.n_user

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __repr__(self):
        fill = max([len(x) for x in self.keys()]) + 2
        return '\n'.join(f'{(k+":").ljust(fill)}{v}' for k, v in self.items())