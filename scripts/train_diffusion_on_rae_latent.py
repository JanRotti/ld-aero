#!/home/jrottmay/.conda/envs/su/bin/python3.10
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
import glob

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
                    prog='LD-Training',
                    description='Trains a Diffusion model on the latent space of a VQVAE model based on a RAE-dataset.',
                    epilog='Work in Progress')

    parser.add_argument('-f', '--force', dest='overwrite', action='store_true', help='Allows training on cpu, deactivated by default.') 
    parser.add_argument('-e', '--emb', dest='emb_dim', type=int, default=1, help='Defines the number of channels for the embedding layer.')
    parser.add_argument('-c', '--chkpt-file', dest='chkpt_file', type=str, default='')
    # END Argparser
    args = parser.parse_args()
    overwrite = args.overwrite
    emb_dim = args.emb_dim
    model_checkpoint = args.chkpt_file

    if not model_checkpoint:
        model_checkpoint = glob.glob('./log/VQVAE-RAE*-model.pth')[-1]

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
            shuffle=True, num_workers=1)
        
    test_data = torch.utils.data.DataLoader(test_split, batch_size=8,
            shuffle=True)

    # Define AE
    ae = VQVAE(1, 32, emb_dim,
        channel_multipliers=(1,2,2,4,4), 
        num_res_blocks=1,
        attention_resolutions=(1, 2, 3),
        num_groups=16,
        norm="gn"
        )
    
    ae.load_state_dict(torch.load(model_checkpoint))

    unet = UNet(
        1,
        16,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        time_emb_dim=None,
        norm="gn",
        num_groups=16,
    )
    betas = generate_linear_schedule(100, 1e-4, 1e-2)

    model = GaussianDiffusion(
        unet,
        (16, 16),
        1,
        betas
    )
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    def loss_function(model, x, y=None):
        z = x
        with torch.no_grad():
            z = ae.encode(z)[0]

        return {'loss': model(z, y=y)} 

    train(model, opt, train_data, loss_function,
        test_data=test_data,
        iterations=10000,
        checkpoint_rate=1000,
        log_rate=100,
        device=device,
        log_to_wandb=True,
        run_name=f"LD-RAE-{datetime.datetime.now().replace(second=0, microsecond=0)}",
        project_name="LD-RAE",
        chkpt_callback=None
    )

    return None

if __name__=="__main__":
    main()  