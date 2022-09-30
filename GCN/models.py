import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from GCN.layers import GCN, SAGPool
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torchmetrics import MatthewsCorrCoef, F1Score, ConfusionMatrix, Accuracy
from pathlib import Path
from typing import Optional, Dict, Any
from pytorch_lightning.loggers import TensorBoardLogger
import os

from torch.utils.data import random_split

# from utils.utilities import get_model_checkpoint
from pytorch_lightning.profilers import SimpleProfiler

from utils.configuration import save_config
import itertools


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args["num_features"]
        self.nhid = args["nhid"]
        self.num_classes = args["num_classes"]
        self.pooling_ratio = args["pooling_ratio"]
        self.dropout_ratio = args["dropout_ratio"]
        self.sample = args["sample_neighbor"]
        self.sparse = args["sparse_attention"]
        self.sl = args["structure_learning"]
        self.lamb = args["lamb"]

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = SAGPool(self.nhid, self.pooling_ratio)
        self.pool2 = SAGPool(self.nhid, self.pooling_ratio)

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
        self.mcc = MatthewsCorrCoef(num_classes=args["num_classes"])
        self.f1 = F1Score(num_classes=args["num_classes"])
        self.mcc = MatthewsCorrCoef(num_classes=args["num_classes"])
        self.acc = Accuracy(num_classes=args["num_classes"])

        self.mcc_single = MatthewsCorrCoef(
            num_classes=args["num_classes"], average=None
        )
        self.f1_single = F1Score(num_classes=args["num_classes"], average=None)
        self.mcc_single = MatthewsCorrCoef(
            num_classes=args["num_classes"], average=None
        )
        self.acc_single = Accuracy(num_classes=args["num_classes"], average=None)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)
        loss = self.loss_fn(out, data.y)
        self.log("train_loss", loss)
        self.log("train_f1", self.f1(out, data.y))
        self.log("train_mcc", self.mcc(out, data.y))
        self.log("train_acc", self.acc(out, data.y))

        # self.log("train_f1_single", self.f1_single(out, data.y))
        # self.log("train_mcc_single", self.mcc_single(out, data.y))
        # self.log("train_acc_single", self.acc_single(out, data.y))
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)
        loss = self.loss_fn(out, data.y)
        self.log("val_loss", loss)
        self.log("val_f1", self.f1(out, data.y))
        self.log("val_mcc", self.mcc(out, data.y))
        self.log("val_acc", self.acc(out, data.y))

        # self.log("val_f1_single", self.f1_single(out, data.y))
        # self.log("val_mcc_single", self.mcc_single(out, data.y))
        # self.log("val_acc_single", self.acc_single(out, data.y))
        return loss

    def test_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)
        loss = self.loss_fn(out, data.y)
        self.log("test_loss", loss)
        self.log("test_f1", self.f1(out, data.y))
        self.log("test_mcc", self.mcc(out, data.y))
        self.log("test_acc", self.acc(out, data.y))
        # self.log("test_confusion_matrix", self.confusion_matrix(out, data.y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.args["lr"]))
        return optimizer


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, args: Dict[str, Any]):
        """Handels all the data loading and preprocessing"""
        super().__init__()
        self.args = args

        self.dataset = TUDataset(
            "data",
            name=self.args["dataset"],
            use_node_attr=True,
        )

        self.args["num_classes"] = self.dataset.num_classes
        self.args["num_features"] = self.dataset.num_features

    def setup(self, stage: Optional[str] = None) -> None:
        """Loads the dataset and splits the dataset into train, val and test"""

        split_train = int(self.dataset.len() * self.args["split_ratio"])
        split_val = int((self.dataset.len() - split_train) * self.args["test_ratio"])
        split_test = self.dataset.len() - split_val - split_train

        self.train_data, self.eval_data, self.test_data = random_split(
            self.dataset,
            [split_train, split_val, split_test],
            generator=torch.Generator().manual_seed(42),
        )

    def update_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update args for datamodule, to extend for number of classes etc."""
        args.update(self.args)
        return args

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_data,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )


class HPT:
    def __init__(self, args):

        self.args = args

        self.datamodule = GraphDataModule(self.args)
        self.args = self.datamodule.update_args(self.args)

        self.search_space = {
            # "lr": [3e-3, 2e-3, 1e-3],
            "dropout_ratio": [0, 0.1, 0.3],
            "pooling_ratio": [0.1, 0.3, 0.5],
            "random_seed": [21, 1337, 2021],
            "structure_learning": [True, False],
        }

        # self.study = optuna.create_study(
        #     sampler=optuna.samplers.GridSampler(self.search_space)
        # )

        self.id = 0  # for saving all models

        for params in self.grid(self.search_space):
            self.args.update(params)
            result = self.objective()

            print(
                """
            --------------------
            Trial {}
            --------------------
            Params: {}
            --------------------
            Result: {}
            --------------------
            """.format(
                    self.id, params, result
                )
            )
        # self.study.optimize(self.objective, n_trials=3 * 3 * 3 * 2, timeout=None)

    def grid(self, d: Dict):
        # loop through all possible combinations of the dictionary d
        for i in itertools.product(*d.values()):
            yield dict(zip(d.keys(), i))

    def objective(self) -> float:
        """Objective function for optuna, modified to just train with grid search"""

        logger = TensorBoardLogger(save_dir="log", name=self.args["experiment_name"])

        self.model = LightModel(self.args).to(self.args["device"])

        trainer = pl.Trainer(
            num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
            accelerator="cuda",
            devices=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
            logger=logger,
            max_epochs=self.args["epochs"],
            # callbacks=callbacks,
            profiler=SimpleProfiler(logger),
            log_every_n_steps=self.args["log_steps"],
        )

        trainer.fit(self.model, self.datamodule)
        Path(self.args["output_path"]).mkdir(parents=True, exist_ok=True)
        _, _ = save_config(self.args)
        torch.save(
            self.model.state_dict(),
            f"{self.args['output_path']}/model_{self.id}.pt",
        )
        self.id += 1

        return trainer.callback_metrics["val_loss"]
