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

    # Argparser
    parser = argparse.ArgumentParser(
                    prog='VQVAE-Training',
                    description='Trains a VQVAE model on the given RAE-dataset.',
                    epilog='Work in Progress')

    parser.add_argument('-f', '--force', dest='overwrite', action='store_true', help='Allows training on cpu, deactivated by default.') 
    parser.add_argument('-e', '--emb', dest='emb_dim', type=int, default=20, help='Defines the number of channels for the embedding layer.')
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

    data  = RAEFlowFieldsDataset(transform=sample_transform)
    
    split = 0.8
    total_length = len(data)
    train_count  = int(split * total_length)
    test_count = total_length - train_count
    train_split, test_split = torch.utils.data.random_split(data, (train_count, test_count))

    train_data = torch.utils.data.DataLoader(train_split, batch_size=8,
            shuffle=True,num_workers=1)
        
    test_data = torch.utils.data.DataLoader(test_split, batch_size=8,
            shuffle=True)
    
    model = VQVAE(1, 32, emb_dim,
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
        iterations=10000,
        checkpoint_rate=1000,
        log_rate=100,
        device=device,
        log_to_wandb=True,
        run_name=f"VQVAE-RAE-{datetime.datetime.now().replace(second=0, microsecond=0)}",
        project_name="VQVAE-RAE",
        chkpt_callback=None
    )

    return None

def visualize_rae_sample(
        model,
        iteration,
        train_data,
        test_data,
        run_name,
        img_dir,
        log_dir,
        wandb_dir,
        log_to_wandb,
        project_name,
        iterations,
        **kwargs
    ):
    x, y = next(iter(train_data))
    b, c, w, h = x.shape
    wl = w // 2**(len(model.channel_multipliers)-1)
    hl = h // 2**(len(model.channel_multipliers)-1)
    z = torch.randn(num_samples, emb_dim, wl, hl)
    z = z.to(device)
    samples = model.decode(z)
    grid_img = torchvision.utils.make_grid(samples, nrow=5, pad_value=2)
    grid_img = grid_img.permute(1, 2, 0)
    plt.imshow(grid_img)
    plt.axis('off')
    plt.text(0.5,0.0,f"Project: {project_name}\nRun: {run_name}\nTime: {datetime.datetime.now().replace(second=0, microsecond=0)}\nIteration: {iteration}\n")

    if log_to_wandb:
        wandb.log({
            "samples": [wandb.Image(sample) for sample in samples]
        })

    plt.savefig(f"{img_dir}/{project_name}-{run_name}-iteration-{iteration}-model.png")

if __name__=="__main__":
    main()