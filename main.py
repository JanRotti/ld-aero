import argparse, os, sys, datetime, glob, importlib, csv, logging, time
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support by default*")
warnings.filterwarnings("ignore", ".*your python command with `srun` like so*")
import numpy as np
import torch
import datetime

import lightning as pl
from lightning.pytorch.cli import LightningCLI

def get_timestamp():
    """Generates a timestamp in the format dd.mm.yyyy-XX:YY"""
    now = datetime.datetime.now()
    return now.strftime("%d.%m.%Y-%H:%M")

class MyLightningCLI(LightningCLI):
    def before_instantiate_classes(self):
        if self.config.subcommand == "fit":
            if self.config.fit.trainer.logger is not None:
                self.config.fit.trainer.logger.init_args.name = get_timestamp()
        return None

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    sys.path.append(os.getcwd())

    cli = MyLightningCLI(save_config_kwargs={"overwrite": True})