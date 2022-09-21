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

# %%
args_dict = take_hp("configs/baseline.yml")

pl.seed_everything(args_dict["random_seed"], workers=True)
# %%
# Load the dataset
dataset = TUDataset(
    Path.cwd().joinpath("data", args_dict["dataset"]),
    name=args_dict["dataset"],
    use_node_attr=True,
)
args_dict["num_classes"] = dataset.num_classes
args_dict["num_features"] = dataset.num_features

# %%
filename, filepath = save_config(args_dict)
# %%
split_train = int(dataset.len() * args_dict["split_ratio"])
split_val = int((dataset.len() - split_train) * args_dict["test_ratio"])
split_test = dataset.len() - split_val - split_train

print(
    f"Hole: {dataset.len()}, Train: {split_train}, Val: {split_val}, Test: {split_test}"
)
# %%
train_data, eval_data, test_data = random_split(
    dataset,
    [split_train, split_val, split_test],
    generator=torch.Generator().manual_seed(42),
)

# %%
train_loader = DataLoader(
    train_data,
    batch_size=args_dict["batch_size"],
    shuffle=True,
    num_workers=args_dict["num_workers"],
)
val_loader = DataLoader(
    eval_data,
    batch_size=args_dict["batch_size"],
    shuffle=False,
    num_workers=args_dict["num_workers"],
)
test_loader = DataLoader(
    test_data,
    batch_size=args_dict["batch_size"],
    shuffle=False,
    num_workers=args_dict["num_workers"],
)

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
        save_dir=args_dict["log_path"], name=args_dict["experiment_name"]
    )
else:
    assert False, "No logger defined"

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
