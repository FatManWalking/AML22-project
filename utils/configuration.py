from __future__ import annotations
import argparse
import yaml
from pathlib import Path
import glob
import torch
from typing import List, Set, Dict, Tuple, Optional

# TODO:
# Train-Test-Split
# Random-Seed

def parse_arguments():

    p = argparse.ArgumentParser(
        description="Train a model for Graph classification"
        "Has Hyperparameter and Logging capabilities to fully customize"
    )

    p.add_argument(
        "--config", type=argparse.FileType(mode="r"), default="configs/baseline.yml",
        help="Every parameter is defined by a config file. See configs/baseline.yml for an example"
    )

    args = p.parse_args()

    print(args)

    if args.config:
        data = yaml.load(
            args.config, Loader=yaml.FullLoader
        )  # Instead of setting every parameter by --<param> when running the script, we can define it all in one yml file
        arg_dict = args.__dict__
        for key, value in data.items():

            arg_dict[key] = value
    
    else:
        raise ValueError("No config file was given")

    if (
        args.device == "cuda" or "auto" and torch.cuda.is_available()
    ):  # Setting up everything needed to run the model on the GPU if possible

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        args.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

    else:
        args.device = torch.device("cpu")
        print("USING CPU BECAUSE GPU IS NOT AVAILABLE")

    return args


def save_config(config: dict, hpt:bool = False) -> Tuple[str, Path]:
    """
    creates directories and config-yml for the current experiment settings
    returns: filename and filepath for logging
    """

    p = Path("log/")
    p = p.joinpath(
        f"{config['architecture']}"
    )  # After this the file structure looks e.g. like this ./log/HGP-Sage/..
    if hpt:
        p.joinpath("hpt")
    

    filename = file_naming(p, config)
    filepath = p / filename
    p.mkdir(parents=True, exist_ok=True)

    # The config argument is a Object the dumper cannot handle, therefor we use a placeholder instead, we dont need it from here on anymore
    config["config"] = f"{filepath}"
    print(f"Saving config to {filepath}")
    with open(f"{str(filepath)}.yml", "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, indent=4
        )  # We dump the currently used Hyperparameters as yml with a version number in the created/accessed folder

    return filename, filepath


def file_naming(path: Path, config: dict) -> str:
    """versions the file and names it according to attributes or the defined name"""

    # Create a string either based on parameters or the explicit experiment name and append a "_V" for Version
    if config['experiment_name_suffix']:
        filename = (
            f"{config['experiment_name']}_{config['experiment_name_suffix']}"
        )
    else:
        filename = (f"{config['experiment_name']}")
    # Search all already existing files with the given name
    file_list = [name for name in glob.glob(f"{path}/{filename}[0-9][0-9][0-9].yml")]

    # If there are already files with that name
    if file_list:

        # sort the list to get the newest version e.g. [file_V001.yml, file_V002.yml, file_V003.yml] pops "file_V003.yml"
        # [-6:-4] subscribes the Version number file_V**003**.yml --> 003. When converted to int we get 3 and can increment that version by 1
        i = int(sorted(file_list).pop()[-6:-4]) + 1
        i = "00" + str(i)
        i = i[-3:]
        filename += i

    else:
        filename += "001"

    # This filename does not include the file ending to generally use it for all files corresponding to this experiment
    # Example Filenames: "No_Pretraining_V002" (for a model with no pretrained weights) or "E10_B40_LR1e-05_V003" (for a model with no specified name)
    return f"{filename}"

