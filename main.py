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
from utils.imports import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from HGP.models import objective

# %%
args_dict = take_hp("configs/reduced_json.yml")

pl.seed_everything(args_dict["random_seed"], workers=True)

# %%
filename, filepath = save_config(args_dict)

# %%
# Define the model
model = LightModel(args_dict).to(args_dict["device"])
# %%
checkpoint_callback = get_model_checkpoint(args_dict)
callbacks = []
if checkpoint_callback:
    callbacks.append(checkpoint_callback)
if args_dict["resume_from_checkpoint"]:
    print(f"loading checkpoint: {args_dict['output_path']}...")
    model.load_from_checkpoint(checkpoint_path=args_dict["output_path"])

if args_dict["logging"]:
    logger = TensorBoardLogger(
        save_dir=args_dict["log_path"], version=1, name="lightning_logs"
    )
else:
    assert False, "No logger defined"

# %%
# TODO: Use a class to ensure data is given


# %%

study.optimize(objective, n_trials=100, timeout=600)
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
# %%
# TODO: add the best params to the model and log them
# %%
trainer = pl.Trainer(
    num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
    accelerator="cuda",
    devices=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
    logger=logger,
    max_epochs=args_dict["epochs"],
    callbacks=callbacks,
    profiler=SimpleProfiler(logger),
    log_every_n_steps=args_dict["log_steps"],
)

# %%
# Train the model ⚡
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# %%
# Test the model [ONLY USE AFTER HPT SEARCH IS DONE]
trainer.test(model, dataloaders=test_loader)

torch.save(model.state_dict(), f"{args_dict['output_path']}/model.pt")
