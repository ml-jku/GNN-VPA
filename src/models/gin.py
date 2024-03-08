import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric
from torch_geometric.nn import global_max_pool
from torch import spmm

from src.tg_vpa.gin_conv import GINConv



class GINModel(pl.LightningModule):
    def __init__(self, num_layers, num_mlp_layers, num_features, hidden_dim, num_classes,
                 final_dropout, train_eps, graph_pooling_type, neighbor_pooling_type,lr):
        super(GINModel, self).__init__()

        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.lr = lr

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.final_dropout = final_dropout
        self.train_eps = train_eps

        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers-1):
            if layer == 0:
                mlp = self.build_mlp(num_mlp_layers, num_features, hidden_dim)
            else:
                mlp = self.build_mlp(num_mlp_layers, hidden_dim, hidden_dim)
            conv = GINConv(nn=mlp, train_eps=train_eps, aggr=self.neighbor_pooling_type)

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(num_features, num_classes))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, num_classes))

        self.loss = nn.CrossEntropyLoss()
        if self.num_classes == 2:
            task = 'binary'
        else:
            task = 'multiclass'
        self.best_val_acc = 0
        self.best_epoch = 0

        self.train_loss = []
        self.step_size = []

        self.metrics = {
            "train": {"acc": Accuracy(task=task, num_classes = self.num_classes),
                      "pool_mean": MeanMetric(),
                      "pool_abs_dev": MeanMetric(),
                      "pool_std": MeanMetric()
                      },
            "valid": {"acc": Accuracy(task=task, num_classes = self.num_classes),
                      "pool_mean": MeanMetric(),
                      "pool_abs_dev": MeanMetric(),
                      "pool_std": MeanMetric()
                      },
            "test": {"acc": Accuracy(task=task, num_classes = self.num_classes),
                      }
        }
    

    def build_mlp(self, num_mlp_layers, num_features, hidden_dim):
        layers = []
        layers.append(nn.Linear(num_features, hidden_dim))
        
        for _ in range(num_mlp_layers-1):
            layers.append(nn.BatchNorm1d((hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        return nn.Sequential(*layers)
    

    def preprocess_graphpool(self, batch):

        '''
        create sum, mean or vpa pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        '''
        
        len_list = torch.bincount(batch)

        idx = []
        elem = []

        start_idx = 0
        for i, graph_len in enumerate(len_list):
            if self.graph_pooling_type in ['sum', 'add']:
                elem.extend([1]*graph_len)
            elif self.graph_pooling_type in ['mean', 'average']:
                elem.extend([1./graph_len]*graph_len)
            elif self.graph_pooling_type in ['vpp', 'vp', 'vpa']:
                elem.extend([1./torch.sqrt(graph_len)]*graph_len)
            elif self.graph_pooling_type == 'max':
                return None
            else:
                raise KeyError(f'{self.graph_pooling_type} is not a valid graph pooling type, chose sum, mean or vpa.')
            
            idx.extend([[i, j] for j in range(start_idx, start_idx + graph_len, 1)])
            start_idx += graph_len
        
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)

        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(len_list), len(batch)]))
        
        return graph_pool.to(self.device)
    

    def forward(self, data):

        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)
   
        graph_pool = self.preprocess_graphpool(batch)

        hiddens = [x]
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            hiddens.append(x)
        
        score_over_layer = 0
        for layer, h in enumerate(hiddens):
            if self.graph_pooling_type == 'max':
                pooled_h = global_max_pool(h, batch)
            else:
                pooled_h = spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        pool_stats = (score_over_layer.mean(), score_over_layer.abs().mean(), score_over_layer.std())

        return score_over_layer, pool_stats

    def process_step(self, batch, batch_idx, mode):
        x, stats = self(batch)
        y = batch.y
        loss = self.loss(x, y)
        if mode =='train':
            self.train_loss.append(loss.detach().cpu().numpy())
            self.step_size.append(len(y))

        self.log(f"{mode}/loss", loss)
        _, preds = torch.max(x, 1)

        self.metrics[mode]["acc"].update(preds, y)

        if mode != 'test':
            self.metrics[mode]["pool_mean"].update(stats[0])
            self.metrics[mode]["pool_abs_dev"].update(stats[1])
            self.metrics[mode]["pool_std"].update(stats[2])

        return loss

    def on_train_epoch_start(self):
        for i in self.metrics["train"]:
            self.metrics["train"][i].to(self.device)

    def on_validation_epoch_start(self):
        for i in self.metrics["valid"]:
            self.metrics["valid"][i].to(self.device)

    def on_test_epoch_start(self):
        for i in self.metrics["test"]:
            self.metrics["test"][i].to(self.device)

    def training_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self.process_step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        self.process_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self.train_acc =  self.metrics["train"]["acc"].compute()
        for i in self.metrics["train"]:
            self.log("train/" + i, self.metrics["train"][i].compute())
            self.metrics["train"][i].reset()
        
    def on_validation_epoch_end(self):
        val_acc = self.metrics["valid"]['acc'].compute()
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = self.current_epoch
        for i in self.metrics["valid"]:
            self.log("valid/" + i, self.metrics["valid"][i].compute())
            self.metrics["valid"][i].reset()
        self.val_acc = val_acc

    def on_test_epoch_end(self):
        test_acc = self.metrics["test"]['acc'].compute()
        self.log("test/acc", test_acc)
        self.test_acc = test_acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

