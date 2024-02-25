# ld-aero
The goal of this repository is research and development towards Latent Diffusion like generative models for design space exploration of aerodynamic datasets. 
The code base is mostly based on adapted versions of existing code repositories with some utility functions. I try to keep the used repositories as updated as possible.

## References
1. [latent-diffusion](https://github.com/CompVis/latent-diffusion) 
2. [Pytorch-VAE](https://github.com/AntixK/PyTorch-VAE)
3. [taming-transformer](https://github.com/CompVis/taming-transformers)

# Comments

## Generative Adversarial Networks
1. Generator, DCGAN: Found checkerboard artifacts on samples.
    - **Problem**: Deconvolutions, often used in the generator to upsample images, can introduce checkerboard patterns due to uneven overlap in the kernel during the upsampling process.
    - **Solutions**:
    Resize and Convolution: Instead of deconvolution, use simpler upsampling techniques like nearest-neighbor or bilinear interpolation, followed by regular convolution layers.
    Kernel Size Adjustments: Experiment with kernel sizes that are divisible by the stride. This ensures better overlap during the upsampling.
    - **Codechange**: Replaced 'ConvTransposed2d' with parameters $(4,2,1)$ with combination of 'Upsample(2)' and inplace 'Conv2d($\cdot,\cdot,3,1,1$).'