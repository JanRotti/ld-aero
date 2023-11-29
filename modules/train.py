import datetime
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm


def train(
        model,
        optimizer,
        train_data,
        loss_function,
        test_data=None,
        iterations=100000,
        checkpoint_rate=10000,
        log_rate=1000,
        device=None,
        log_to_wandb=False,
        model_checkpoint=None,
        optim_checkpoint=None,
        run_name="testing",
        wandb_dir="./tmp",
        img_dir="./img",
        log_dir="./log",
        entity='jan-rottmayer',
        project_name="unnamed",
        train_callback=None,
        chkpt_callback=None
    ):

    # Determine current device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model = model.to(device)

    # Load chckp
    if model_checkpoint is not None:
        model.load_state_dict(torch.load(model_checkpoint))
    if optim_checkpoint is not None:
        optimizer.load_state_dict(torch.load(optim_checkpoint))

   # Start wandb log
    if log_to_wandb:
        run = wandb.init(
                        project=project_name,
                        dir=wandb_dir,
                        entity=entity,
                        name=run_name,
                    )
        wandb.watch(model)

    # Initialize loop variables
    train_loss = 0
    test_loss = 0 
    iteration = 0

    pbar = tqdm(range(1, iterations + 1), postfix=f"")
    for iteration in pbar:
        # Set model to 'train' mode
        model.train()

        # Retreive random batch -> assumes sample & label return
        x, y = next(iter(train_data))
        x = x.to(device)
        y = y.to(device)

        # Compute loss
        loss = loss_function(model, x, y)
        train_loss += loss['loss'].item()

        # Perform backprop
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        # callback
        if train_callback is not None:
            train_callback()

        # Logging
        if iteration % log_rate == 0:
            test_loss = 0
            if test_data is None:
                test_loss = np.nan
            else:
                with torch.no_grad():
                    model.eval()
                    for x, y in iter(test_data):
                        x = x.to(device)
                        y = y.to(device)

                        loss = loss_function(model, x, y)
                        test_loss += loss['loss'].item()

                test_loss /= len(test_data)
            # Get mean batch loss
            train_loss /= log_rate 
            # Log to tqdm pbar
            pbar.set_postfix({"Train Loss": train_loss, "Test Loss": test_loss})
            # log to wandb
            if log_to_wandb:
                wandb.log({
                            "test_loss": test_loss,
                            "train_loss": train_loss,
                            **loss
                        })
            
            # Reset train loss
            train_loss = 0

        # Checkpointing
        if iteration % checkpoint_rate == 0:

            # Define files
            model_filename = f"{log_dir}/{project_name}-{run_name}-iteration-{iteration}-model.pth"
            optim_filename = f"{log_dir}/{project_name}-{run_name}-iteration-{iteration}-optim.pth"
            # Save locally
            torch.save(model.state_dict(), model_filename)
            torch.save(optimizer.state_dict(), optim_filename)

            if chkpt_callback is not None:
                chkpt_callback()

    return None

    