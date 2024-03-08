import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation
from src.agg import VariancePreservingAggregation

from src.tg_vpa.sg_conv import SGConv


class SGCModel(pl.LightningModule):
    def __init__(self, num_layers, num_features, conv_hidden_dim, fc_hidden_dim, num_classes,
                 final_dropout, graph_pooling_type, neighbor_pooling_type, lr):
        super(SGCModel, self).__init__()

        self.num_layers = num_layers
        self.lr = lr

        self.num_features = num_features
        self.conv_hidden_dim = conv_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.num_classes = num_classes

        self.final_dropout = final_dropout

        self.neighbor_pooling_type = neighbor_pooling_type
        self.graph_pooling_type = graph_pooling_type

        if self.graph_pooling_type in ['sum', 'add', 'default']:
            self.pool = SumAggregation()
        elif self.graph_pooling_type in ['mean', 'average']:
            self.pool = MeanAggregation()
        elif self.graph_pooling_type in ['vpa', 'vpp', 'vp']:
            self.pool = VariancePreservingAggregation()

        self.conv = SGConv(self.num_features, self.conv_hidden_dim, self.num_layers, aggr=self.neighbor_pooling_type)

        self.fc1 = nn.Linear(self.conv_hidden_dim, self.fc_hidden_dim)
        self.fc2 = nn.Linear(self.fc_hidden_dim, self.num_classes)

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
            "train": {"acc": Accuracy(task=task, num_classes=self.num_classes),
                      "pool_mean": MeanMetric(),
                      "pool_abs_dev": MeanMetric(),
                      "pool_std": MeanMetric()
                      },
            "valid": {"acc": Accuracy(task=task, num_classes=self.num_classes),
                      "pool_mean": MeanMetric(),
                      "pool_abs_dev": MeanMetric(),
                      "pool_std": MeanMetric()
                      },
            "test": {"acc": Accuracy(task=task, num_classes=self.num_classes),
                     }
        }

    def forward(self, data):

        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)

        x = F.relu(self.conv(x, edge_index))

        x = self.pool(x, batch)

        x = F.dropout(x, p=self.final_dropout, training=self.training)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        pool_stats = (x.mean(), x.abs().mean(), x.std())

        return x, pool_stats

    def process_step(self, batch, batch_idx, mode):
        x, stats = self(batch)
        y = batch.y
        loss = self.loss(x, y)

        if mode == 'train':
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
        self.train_acc = self.metrics["train"]["acc"].compute()
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

