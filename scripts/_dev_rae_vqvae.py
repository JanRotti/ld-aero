#!/home/jrottmay/.conda/envs/su/bin/python3.10
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
from scipy.ndimage import zoom

from modules import *

def sample_transform(sample):
    # Make sample square
    w, h, c = sample.shape
    cut = min(w, h) 
    # Use single channel
    sample = sample[:cut,:cut,0]
    # Desired dim
    ddim = 256
    zoom_factor = ddim / cut
    # Downsample 
    sample = zoom(sample, zoom_factor)
    # Add channel dim back
    sample = np.expand_dims(sample, 0)
    # Convert to torch tensor
    sample = torch.from_numpy(sample)
    # Convert from double to single prec
    sample = sample.type(torch.float)
    return sample

def main():
    overwrite = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Additional Info when using cuda
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    if device == "cpu" and not overwrite:
        print("No model training performed. 'cuda' not available, and training is to expensive to run on cpu.")
        return None

    data  = RAEFlowFieldsDataset(transform=sample_transform)
    w, h, c = data[0][0].shape
    
    split = 0.8
    total_length = len(data)
    train_count  = int(split * total_length)
    test_count = total_length - train_count
    train_split, test_split = torch.utils.data.random_split(data, (train_count, test_count))

    train_data = torch.utils.data.DataLoader(train_split, batch_size=8,
            shuffle=True,num_workers=1)
        
    test_data = torch.utils.data.DataLoader(test_split, batch_size=8,
            shuffle=True)
    
    model = VQVAE(1, 32, 100,
        channel_multipliers=(1,2,2,4,4), 
        num_res_blocks=1,
        attention_resolutions=(1, 2, 3),
        num_groups=16,
        norm="gn"
        )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_function = lambda m,x,y: m.loss(x, y=y)
    
    train(model, opt, train_data, loss_function,
        test_data=test_data,
        iterations=10
    )

    return None


if __name__=="__main__":
    main()