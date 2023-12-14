import datetime
import torchvision
import matplotlib.pyplot as plt

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

def vqvae_chkpt_callback(
    model,
    train_data=None,
    project_name=None,
    run_name=None,
    iteration=None,
    **kwargs
    ):
    
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No need to calculate gradients for this
        for samples, y in vis_samples_loader:
            # Assuming your DataLoader returns just images, adjust if it returns a tuple (images, labels) or similar
            reconstructions = model(samples.to(device)).to('cpu')

            # Concatenate original images and reconstructions
            comparison = torch.cat([samples, reconstructions])
            comparison = comparison[:,:1,:,:]
            
            # Create a grid of images
            grid = torchvision.utils.make_grid(comparison, nrow=n_samples, padding=2, normalize=True)

            # Convert the tensor to a PIL image and display it
            plt.imshow(grid.permute(1, 2, 0).numpy())
            plt.text(0.5,0.0,f"Project: {project_name}\nRun: {run_name}\nTime: {datetime.datetime.now().replace(second=0, microsecond=0)}\nIteration: {iteration}\n")
            plt.axis('off')
            plt.show()
        
    model.train()
    return None