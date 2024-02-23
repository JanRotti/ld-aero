import torch
import os
import importlib
import argparse
from torchsummary import summary
import lightning.pytorch as pl
import yaml

def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Summary")
    parser.add_argument("-c", "--config", dest="config", type=str, help="Path to the config file", required=True)
    parser.add_argument("-i", "--input_shape", dest="input_shape", nargs='*', type=int, help="Input shape as tuple", required=True, default=[1, 28, 28])
    args = parser.parse_args()

    # Load the config file
    config_file = args.config
    input_shape = tuple(args.input_shape)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Load the model from the config
    model = instantiate_from_config(config["model"])

    # Generate the model summary
    summary(model, input_size=input_shape)