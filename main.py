import argparse, os, sys, datetime, glob, importlib, csv, logging, time
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support by default*")
warnings.filterwarnings("ignore", ".*your python command with `srun` like so*")
import numpy as np
import torch

import lightning as pl
from lightning.pytorch.cli import LightningCLI

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    sys.path.append(os.getcwd())

    cli = LightningCLI(save_config_kwargs={"overwrite": True})