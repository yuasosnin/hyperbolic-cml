from typing import *
from numpy.typing import ArrayLike

import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sklearn.manifold import TSNE
from umap.umap_ import UMAP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geoopt

from .utils import *


class MetricTrainer(nn.Module):
    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = None
        self.scheduler = None
        self.logs = {}

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.model.forward(users, items)
    
    def log(self, name: str, value: Union[float, int, np.array]) -> None:
        if name not in self.logs:
            self.logs[name] = np.array([])
        if isinstance(value, (float, int)):
            self.logs[name] = np.append(self.logs[name], value)
        elif isinstance(value, np.ndarray):
            self.logs[name] = np.concatenate(self.logs[name], value)

    def _epoch_reduce(self, name: str, loader_len: int, reduction: Literal['mean', 'sum'] = 'mean') -> np.array:
        if reduction == 'mean':
            return self.logs[name][-loader_len:].sum() / loader_len
        elif reduction == 'sum':
            return self.logs[name][-loader_len:].sum()
    
    def training_step(self, batch: Tuple[torch.Tensor]) -> torch.Tensor:
        users, items, neg_items = batch
        users, items, neg_items = users.to(self.device), items.to(self.device), neg_items.to(self.device)

        user_embedding, pos_item_embedding, neg_item_embedding = self.model(users, items, neg_items)

        loss = self.criterion(user_embedding, pos_item_embedding, neg_item_embedding)
        self.log('train_step_loss', loss.item())
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor]) -> torch.Tensor:
        users, pos_items, neg_items = batch
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        user_embedding, pos_item_embedding, neg_item_embedding = self.model(users, pos_items, neg_items)

        loss = self.criterion(user_embedding, pos_item_embedding, neg_item_embedding)
        self.log('valid_step_loss', loss.item())

        pos_item_dist, neg_item_dist = self.model.predict(users, pos_items, neg_items)
        all_items = torch.cat([pos_items, neg_items], dim=1).detach()
        all_item_dist = torch.cat([pos_item_dist, neg_item_dist], dim=1).detach()

        hits, size = hit_rate(pos_items, all_items, all_item_dist)
        cg, size = ndcg(pos_items, all_items, all_item_dist)
        self.log('valid_step_hits', hits)
        self.log('valid_step_ndcg', cg)
        self.log('valid_step_size', size)
        return loss
    
    def fit(
        self,
        train_loader: DataLoader, 
        valid_loader: DataLoader, 
        test_tensor: Optional[torch.Tensor] = None, 
        epochs: int = 3, 
        plot: bool = False, 
        validate_every: int = 1
    ) -> None:
        
        for epoch in tqdm(range(epochs)):
            print('epoch:', epoch+1)

            self.model.train()
            for batch in tqdm(train_loader):
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.log(
                'train_epoch_loss', 
                self._epoch_reduce('train_step_loss', len(train_loader)))
            print('train_loss:', self.logs['train_epoch_loss'][-1])
            
            if (epoch+1) % validate_every == 0:
                self.model.eval()
                for batch in tqdm(valid_loader):
                    loss = self.validation_step(batch)
                self.log(
                    'valid_epoch_loss', 
                    self._epoch_reduce('valid_step_loss', len(valid_loader)))
                print('valid_loss:', self.logs['valid_epoch_loss'][-1])

                total_hits = self._epoch_reduce('valid_step_hits', len(valid_loader), reduction='sum')
                total_ndcg = self._epoch_reduce('valid_step_ndcg', len(valid_loader), reduction='sum')
                total_size = self._epoch_reduce('valid_step_size', len(valid_loader), reduction='sum')
                self.log(
                    'valid_epoch_hitrate', 
                    total_hits/total_size)
                self.log(
                    'valid_epoch_ndcg',
                    total_ndcg/total_size)
                print('valid_hitrate:', f'{int(total_hits)}/{int(total_size)}')
                print('valid_ndcg:', f'{round(total_ndcg, 2)}/{int(total_size)}')
            
            if plot:
                self.plot_embeddings()
                
            print('-'*80)
    
    def predict_embeddings(self, project_onto_euclidean: bool = False) -> Tuple[torch.Tensor]:
        self.model.eval()
        all_users = torch.tensor(list(range(self.model.n_user))).to(self.device)
        all_items = torch.tensor(list(range(self.model.n_item))).to(self.device)
        user_embeddings = self.model.user_embedding(all_users).detach()
        item_embeddings = self.model.item_embedding(all_items).detach()
        if project_onto_euclidean:
            return self.model.manifold.logmap0(user_embeddings), self.model.manifold.logmap0(item_embeddings)
        return user_embeddings, item_embeddings
    
    def predict_lists(self) -> torch.Tensor:
        user_emb, item_emb = self.predict_embeddings()
        out = torch.zeros((self.model.n_user, self.model.n_item))
        for i in range(self.model.n_user):
            out[i, :] = torch.cdist(user_emb[i, :].view(1, -1), item_emb).flatten()
        return out
    
    def plot_embeddings(
        self, 
        ax=None,
        algorithm: Literal['TSNE', 'UMAP'] = 'TSNE', 
        item_sizes: Optional[ArrayLike] = None, 
        user_sizes: Optional[ArrayLike] = None, 
        project_onto_euclidean: bool = False,
        item_kwargs: dict = {},
        user_kwargs: dict = {}
    ) -> plt.Axes:
        
        user_emb, item_emb = self.predict_embeddings(project_onto_euclidean=project_onto_euclidean)
        emb = torch.cat([user_emb.cpu(), item_emb.cpu()], dim=0)
        
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if algorithm in {'TSNE', 'tsne'}:
            mapped_emb = TSNE(metric='euclidean', init='pca').fit_transform(emb)
            mapped_user_emb, mapped_item_emb = mapped_emb[:self.model.n_user, :], mapped_emb[self.model.n_user:, :]
        elif algorithm in {'UMAP', 'umap'}:
            mapped_emb = UMAP(metric='euclidean', n_neighbors=30).fit_transform(emb)
            mapped_user_emb, mapped_item_emb = mapped_emb[:self.model.n_user, :], mapped_emb[self.model.n_user:, :]
        else:
            raise ValueError('Algorithm not supported')
            
        if ax is None:
            ax = plt.gca()
            
        ax.scatter(mapped_item_emb[:, 0], mapped_item_emb[:, 1], s=item_sizes, **(dict(alpha=0.5, color='blue') | item_kwargs))
        ax.scatter(mapped_user_emb[:, 0], mapped_user_emb[:, 1], s=user_sizes, **(dict(alpha=0.5, color='red') | user_kwargs))
        return ax
        
    def plot_loss(self) -> plt.Axes:
        raise NotImplemented
