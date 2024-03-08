import pytorch_lightning as pl
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset


import os
import random


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True


def tag2index(dataset):
    tag_set = []
    for g in dataset:
        all_nodes = torch.cat([g.edge_index[0], g.edge_index[1]])
        node_tags = torch.bincount(all_nodes, minlength=g.num_nodes)/2
        node_tags = list(set(list(np.array(node_tags))))
        tag_set += node_tags
    tagset = list(set(tag_set))
    tag2index_dict = {int(tagset[i]):i for i in range(len(tagset))}
    return tag2index_dict


def get_splits(dataset, seed, fold_idx):

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    idx_list = []

    for idx in kfold.split(np.zeros(len(dataset.y)), dataset.y):
        idx_list.append(idx)

    train_val_idx, test_idx = idx_list[fold_idx]

    train_idx, val_idx = train_test_split(train_val_idx, train_size=0.888889, random_state=seed, stratify=dataset.y[train_val_idx])

    return train_idx, val_idx, test_idx


def subset_from_indices(dataset, indices):
    return [dataset[i] for i in indices]


class CustomTUDataset(InMemoryDataset):
    def __init__(self, dataset_name, fold_idx, mode, use_node_attr=True, seed=42, deg_features=0):
        dataset = TUDataset(f"./data/", dataset_name, use_node_attr=use_node_attr)

        train_idx, val_idx, test_idx = get_splits(dataset, seed=seed, fold_idx=fold_idx)

        if deg_features == 1:  # set node features to the node's degree

            tag2index_dict = tag2index(dataset)
            processed_dataset = []

            for i in range(len(dataset)):
                g = dataset[i]
                all_nodes = torch.cat([g.edge_index[0], g.edge_index[1]])
                node_tags = list(np.array(torch.bincount(all_nodes, minlength=g.num_nodes)/2))
                features = torch.zeros(g.num_nodes, len(tag2index_dict))
                features[[range(g.num_nodes)], [tag2index_dict[tag] for tag in node_tags]] = 1
                g['x'] = features
                processed_dataset.append(g)
            
            dataset = processed_dataset

        elif dataset_name in ['IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:  # set node features to 1 for datasets that do not contain node features
            processed_dataset = []
            for i in range(len(dataset)):
                g = dataset[i]
                features = torch.ones((g.num_nodes, 1))
                g['x'] = features
                processed_dataset.append(g)

            dataset = processed_dataset

        if mode == "train":
            self.dataset = subset_from_indices(dataset, train_idx)
        elif mode == "test":
            self.dataset = subset_from_indices(dataset, test_idx)
        else:
            self.dataset = subset_from_indices(dataset, val_idx)
        super().__init__(f"./data/{dataset_name}")
        self.data, self.slices = self.collate(self.dataset)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader
    ):
        super().__init__()
        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader
        self.test_dataloader_ = test_dataloader

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_
    
    def test_dataloader(self):
        return self.test_dataloader_




