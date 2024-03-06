# ld-aero
The goal of this repository is research and development towards Latent Diffusion like generative models for design space exploration of aerodynamic datasets. 
The code base is mostly based on adapted versions of existing code repositories with some utility functions. I try to keep the used repositories as updated as possible.

## References
1. [latent-diffusion](https://github.com/CompVis/latent-diffusion) 
2. [Pytorch-VAE](https://github.com/AntixK/PyTorch-VAE)
3. [taming-transformer](https://github.com/CompVis/taming-transformers)
4. [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

# Comments

## Latent Diffusion Model
1. VQVAE and general training: Found OOM error when running on GPU node on cluster
    - **Problem**: Unclear yet, assumption either gradient accumulation (memory leak) or too little accessible RAM
    - **Solution**: Default RAM setting for slurm jobs is 1.8 GB RAM, which might be too little. Adding '#SBATCH --mem=8000' for 8GB should alleviate the problem.  
    - **Codechange**: None! 

## Generative Adversarial Networks
1. Generator, DCGAN: Found checkerboard artifacts on samples.
    - **Problem**: Deconvolutions, often used in the generator to upsample images, can introduce checkerboard patterns due to uneven overlap in the kernel during the upsampling process.
    - **Solutions**:
    Resize and Convolution: Instead of deconvolution, use simpler upsampling techniques like nearest-neighbor or bilinear interpolation, followed by regular convolution layers.
    Kernel Size Adjustments: Experiment with kernel sizes that are divisible by the stride. This ensures better overlap during the upsampling.
    - **Codechange**: Replaced 'ConvTransposed2d' with parameters $(4,2,1)$ with combination of 'Upsample(2)' and inplace 'Conv2d($\cdot,\cdot,3,1,1$).'


## nn.Sequential
1. Using nn.Sequential to wrap layers and blocks into respective blocks does not work due to a restrictive forward pass. nn.Sequential does not allow passing of multiple input arguments. TODO: fix problem with modules.diffusion.unet