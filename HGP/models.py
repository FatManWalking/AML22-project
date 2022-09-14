import torch
from torch_geometric.nn import GCNConv
from HGP.layers import GCN, HGPSLPool
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torchmetrics import MatthewsCorrCoef, F1Score, ConfusionMatrix


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args["num_features"]
        self.nhid = args["nhid"]
        self.num_classes = args["num_classes"]
        self.pooling_ratio = args["pooling_ratio"]
        self.dropout_ratio = float(args["dropout_ratio"])
        self.sample = args["sample_neighbor"]
        self.sparse = args["sparse_attention"]
        self.sl = args["structure_learning"]
        self.lamb = args["lamb"]

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(
            self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb
        )
        self.pool2 = HGPSLPool(
            self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb
        )

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class LightModel(pl.LightningModule):
    def __init__(self, args):
        super(LightModel, self).__init__()
        self.args = args
        self.model = Model(args)
        self.loss_fn = torch.nn.NLLLoss()

        # LOGS
        self.train_mcc = MatthewsCorrCoef(num_classes=args["num_classes"])
        self.train_f1 = F1Score(num_classes=args["num_classes"])
        self.train_confusion_matrix = ConfusionMatrix(num_classes=args["num_classes"])

        self.val_mcc = MatthewsCorrCoef(num_classes=args["num_classes"])
        self.val_f1 = F1Score(num_classes=args["num_classes"])
        self.val_confusion_matrix = ConfusionMatrix(num_classes=args["num_classes"])

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)
        loss = self.loss_fn(out, data.y)
        self.log("train_loss", loss)
        self.log("train_f1", self.train_f1(out, data.y))
        self.log("train_mcc", self.train_mcc(out, data.y))
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)
        loss = self.loss_fn(out, data.y)
        self.log("val_loss", loss)
        self.log("val_f1", self.val_f1(out, data.y))
        self.log("val_mcc", self.val_mcc(out, data.y))
        return loss

    def test_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)
        loss = self.loss_fn(out, data.y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.args["lr"]))
        return optimizer
