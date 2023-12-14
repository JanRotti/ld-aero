#!/home/jrottmay/.conda/envs/su/bin/python3.10
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision

from scipy.ndimage import zoom
from skimage.measure import block_reduce

from modules import *

def transform(sample):
    # Make sample square
    w, h, c = sample.shape
    cut = min(w, h) 
    # Use all channels
    sample = sample[:cut,:cut,:] 
    # Desired dim
    ddim = 128
    zoom_factor = ddim / cut
    # Downsample 
    channels = []
    for i in range(c):
        channel = zoom(sample[:, :, i], zoom_factor)
        channels.append(channel)
    sample = np.stack(channels, axis=-1)
    # Permute dimensions
    sample = sample.transpose(2,0,1)
    # Convert to torch tensor
    sample = torch.from_numpy(sample)
    # Convert from double to single prec
    sample = sample.type(torch.float)
    return sample

def main():

    # Argparser
    parser = argparse.ArgumentParser(
                    prog='VQVAE-Training',
                    description='Trains a VQVAE model on the given RAE-dataset.',
                    epilog='Work in Progress')

    parser.add_argument('-f', '--force', dest='overwrite', action='store_true', help='Allows training on cpu, deactivated by default.') 
    parser.add_argument('-e', '--emb', dest='emb_dim', type=int, default=1, help='Defines the number of channels for the embedding layer.')
    # END Argparser
    args = parser.parse_args()
    overwrite = args.overwrite
    emb_dim = args.emb_dim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Additional Info when using cuda
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    if device == "cpu" and not overwrite:
        print("No model training performed. 'cuda' is not available!")
        return None

    data  = RAEFlowFieldsDataset(transform=transform)
    
    split = 0.8
    total_length = len(data)
    train_count  = int(split * total_length)
    test_count = total_length - train_count
    train_split, test_split = torch.utils.data.random_split(data, (train_count, test_count))

    train_data = torch.utils.data.DataLoader(train_split, batch_size=8,
            shuffle=True,num_workers=1)
        
    test_data = torch.utils.data.DataLoader(test_split, batch_size=8,
            shuffle=True)
    
    n_samples = 5
    subset = torch.utils.data.Subset(train_data.dataset,np.random.randint(0,len(train_data.dataset), size=(n_samples)))
    vis_samples_loader = torch.utils.data.DataLoader(subset, batch_size=n_samples, shuffle=False)
    base_channels = 16
    channel_multipliers = (1,2,2,4,4) 

    model = VQVAE(
        8,
        base_channels,
        1, 
        num_embeddings=20,
        channel_multipliers=channel_multipliers,
        attention_resolutions=(1,2,3),
        num_res_blocks=1,
        time_emb_dim=None,
        dropout=0.0,
        norm="gn",
        num_groups=16,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_function = lambda m,x,y: m.loss(x, y=y)

    train(model, opt, train_data, loss_function,
        test_data=test_data,
        iterations=10000,
        checkpoint_rate=1000,
        log_rate=100,
        device=device,
        log_to_wandb=True,
        run_name=f"{datetime.datetime.now().replace(second=0, microsecond=0)}",
        project_name="VQVAE-RAE",
        chkpt_callback=vqvae_chkpt_callback
    )

    return None

if __name__=="__main__":
    main()