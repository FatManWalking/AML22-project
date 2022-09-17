# -*- coding: utf-8 -*-
# %% [markdown]
# # AML - Final Prtoject - 2022
#
# It is possible to use this as notebook or directly as a script
#
# This notebook is organized in
# * [Configuration for Model and Logging](#config)
# * [Loading Dataset](#dataset)
# * [Model Definition](#model)
# * [Train Model](#train)
# %% [markdown]
# ## Imports
# %%
# %load_ext autoreload
# %autoreload 2

# %%
import imp
from utils.imports import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from HGP.models import HPT

# %%
args_dict = take_hp("configs/reduced_json.yml")

pl.seed_everything(args_dict["random_seed"], workers=True)

# %%
hpt_search = HPT(args_dict)
# %%
study = hpt_search.study
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
trial = study.best_trial

# %%
hpt_search.datamodule.update_args(args_dict)
filename, filepath = save_config(args_dict)

# %%
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
