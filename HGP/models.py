import torch
from torch_geometric.nn import GCNConv
from HGP.layers import GCN, HGPSLPool
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torchmetrics import MatthewsCorrCoef, F1Score, ConfusionMatrix

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from pathlib import Path
from typing import Optional

from torch.utils.data import random_split


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


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, args: dict):
        """Handels all the data loading and preprocessing"""
        super().__init__()
        self.args = args

    def setup(self, stage: Optional[str] = None) -> None:
        """Loads the dataset and splits the dataset into train, val and test"""
        dataset = TUDataset(
            str(Path.cwd().joinpath("data", self.args["dataset"])),
            name=self.args["dataset"],
            use_node_attr=True,
        )

        self.args["num_classes"] = dataset.num_classes
        self.args["num_features"] = dataset.num_features

        split_train = int(dataset.len() * self.args["split_ratio"])
        split_val = int(
            dataset.len() * 1 - self.args["split_ratio"] / self.args["test_ratio"]
        )
        split_test = dataset.len() - split_val - split_train

        self.train_data, self.eval_data, self.test_data = random_split(
            dataset,
            [split_train, split_val, split_test],
            generator=torch.Generator().manual_seed(42),
        )

    def update_args(self, args):
        """Update args for datamodule, to extend for number of classes etc."""
        return args.update(self.args)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
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

        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner(n_startup_trials=10)
            if args["pruning"]
            else optuna.pruners.NopPruner()
        )
        self.study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
        )

        self.study.optimize(self.objective, n_trials=100, timeout=600)

    def objective(self, trial: optuna.trial.Trial) -> float:

        # TODO: actual use layer params

        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        output_dims = [
            trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
            for i in range(n_layers)
        ]
        hpt_dict = {
            "nhid": trial.suggest_int("nhid", 64, 256),
            "dropout_ratio": trial.suggest_float("dropout_ratio", 0, 0.01),
            "pooling_ratio": trial.suggest_float("pooling_ratio", 0, 0.5),
            "sample": trial.suggest_categorical("sample", [True, False]),
            "sparse": trial.suggest_categorical("sparse", [True, False]),
            "lambs": trial.suggest_float("lambs", 0.5, 1),
            "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 5),
            "structure_learning": trial.suggest_categorical(
                "structure_learning", [True, False]
            ),
        }

        self.args.update(hpt_dict)

        model = LightModel(self.args).to(self.args["device"])
        datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

        trainer = pl.Trainer(
            logger=True,
            limit_val_batches=PERCENT_VALID_EXAMPLES,
            enable_checkpointing=False,
            max_epochs=EPOCHS,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        )
        hyperparameters = dict(
            n_layers=n_layers, dropout=dropout, output_dims=output_dims
        )
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        return trainer.callback_metrics["val_acc"].item()
