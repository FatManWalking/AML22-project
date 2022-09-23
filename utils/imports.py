"""
This module centralizes all imports that are needed across all of the main scripts
because they are always the same and it is easier to maintain imports this why when module paths change

It also helps to ensure all imports succeed when running the scripts and wont fail because of import errors
"""

from pathlib import Path
import os
import numpy as np
import math

# import pandas as pd
import yaml

# import matplotlib.pyplot as plt
# import json
import sys
from IPython import get_ipython
from typing import List, Set, Dict, Tuple, Optional

# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter  # Path changed in newer version
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce  # Coalesce is not backward compatible

from torchmetrics import MatthewsCorrCoef, F1Score, ConfusionMatrix


from HGP.models import Model, LightModel
from HGP.sparse_softmax import Sparsemax
from HGP.layers import GCN, HGPSLPool
from utils.configuration import parse_arguments, save_config
from utils.utilities import get_model_checkpoint
from utils.parameters import take_hp
