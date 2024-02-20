import argparse, os, sys, datetime, glob, importlib, csv, logging, time
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import numpy as np
import torch

import lightning as pl
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning import seed_everything

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    sys.path.append(os.getcwd())

    cli = LightningCLI(save_config_kwargs={"overwrite": True})