from utils.configuration import parse_arguments
import yaml
from IPython import get_ipython
import json

from typing import List, Set, Dict, Tuple, Optional


def yaml_dump_for_notebook(filepath="configs/baseline.yml"):
    # Call this function if you use this as a notebook so bypass argparse and directly dump the config
    with open(str(filepath), "r", encoding="utf-8") as stream:
        args_dict = yaml.safe_load(stream) or dict()

    return args_dict


def isnotebook():
    """
    a simple function to quickly check if something is running in a notebook

    This is usefull to run the repository as a notebook (just dumping the yml in a dict) as well as a script (with the argparser)
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def take_hp(config: str) -> dict:
    """
    Takes a string path and returns a dict with parameters found in the yaml at the specified location
    """
    if isnotebook():
        # In a notebook you need to just dump the yaml with the configuration details
        args_dict = yaml_dump_for_notebook(filepath=config)
        # print(args_dict)
    else:
        # This can only be used if this is run as a script. For notebooks use the yaml.dump and configure the yaml file accordingly
        args = parse_arguments()
        args_dict = args.__dict__

    args_dict["output_path"] = (
        args_dict["output_path"] + "/" + args_dict["experiment_name"]
    )
    args_dict["log_path"] = args_dict["log_path"] + "/" + args_dict["experiment_name"]
    return args_dict
