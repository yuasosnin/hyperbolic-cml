import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import geoopt
from geoopt.optim import RiemannianAdam

from models import *
from losses import *
from utils import *
from trainer import MetricTrainer
from dataset import PariwiseDataset

DATA = 'samokat'
SIZE = '100k'


if DATA == 'movielens':
    data = load_data(f'ml-{SIZE}')
    data_train, data_valid, data_test = train_test_split_single(data, how='last')
elif DATA == 'samokat':
    data = pd.read_csv(f'sk/data_{SIZE}.csv')
    data_train, data_valid, data_test = train_test_split_single(data, how='random')
    
    
N_USERS = data.user_id.nunique()
N_ITEMS = data.item_id.nunique()
user_sizes = data.groupby('user_id').count().item_id.to_numpy()
item_sizes = data.groupby('item_id').count().user_id.to_numpy()

it = iter(range(9999))

def run(cfg, mode='valid', model='CML', epochs=50, num_workers=4):
    
    
    train_set = PariwiseDataset(
        data_train, 
        N_USERS, 
        N_ITEMS, 
        neg_samples=cfg.neg_samples, 
        weights=(item_sizes if cfg.weighted else None))

    valid_set = PariwiseDataset(data_valid, N_USERS, N_ITEMS, neg_samples=100)
    test_set = PariwiseDataset(data_test, N_USERS, N_ITEMS, neg_samples=100)

    train_loader = DataLoader(train_set, batch_size=cfg.bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=N_USERS, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=N_USERS, shuffle=False)


    if model == 'CML':
        metric_model = CML(N_USERS, N_ITEMS, embedding_dim=cfg.embedding_dim, dropout_rate=cfg.drop_rate, max_norm=cfg.max_norm)
        metric_criterion = AdditiveLoss(
            nn.TripletMarginLoss(margin=cfg.margin),
            CovarianceLoss(),
            coefficients=[1, cfg.lam]
        )

        metric_trainer = MetricTrainer(metric_model, metric_criterion)
        metric_trainer.optimizer = Adam(metric_trainer.model.parameters(), lr=cfg.lr)
    elif model == 'HyperCML':
        manifold = geoopt.PoincareBall(c=1, learnable=True)
        metric_model = HyperCML(
            N_USERS, 
            N_ITEMS, 
            embedding_dim=cfg.embedding_dim, 
            dropout_rate=cfg.drop_rate,
            manifold=manifold, 
            max_norm=cfg.max_norm)
        metric_criterion = AdditiveLoss(
            nn.TripletMarginWithDistanceLoss(margin=cfg.margin, distance_function=manifold.dist),
            DistortionLoss(manifold=manifold),
            coefficients=[1, cfg.lam]
        )

        metric_trainer = MetricTrainer(metric_model, metric_criterion)
        metric_trainer.optimizer = RiemannianAdam(metric_trainer.model.parameters(), lr=cfg.lr)

    if mode == 'valid':
        metric_trainer.fit(train_loader, valid_loader, epochs=epochs, plot=False, validate_every=50)
    elif mode == 'test':
        metric_trainer.fit(train_loader, test_loader, epochs=epochs, plot=False, validate_every=50)
    
    fig = metric_trainer.plot_embeddings(item_sizes=item_sizes, user_sizes=user_sizes, algorithm='UMAP')
    figname = next(it)
    plt.savefig(f'images/{mode}/{figname}')
    if mode == 'valid':
        full_hr = full_hit_rate(metric_trainer, valid_set, data_train)
    elif mode == 'test':
        full_hr = full_hit_rate(metric_trainer, test_set, data_train)
    
    row = ','.join(map(str, 
        [
            model, 
            cfg.embedding_dim, 
            cfg.margin, 
            cfg.lam, 
            cfg.lr,
            metric_trainer.logs['valid_epoch_loss'][-1], 
            int(metric_trainer.logs['valid_step_hits'][-1]), 
            metric_trainer.logs['valid_epoch_hitrate'][-1],
            metric_trainer.logs['valid_epoch_ndcg'][-1],
            full_hr,
            figname
        ]
    ))

    with open(f'logs_{mode}.txt', 'a') as f:
        f.write(row)
        f.write('\n')
        
    return metric_trainer