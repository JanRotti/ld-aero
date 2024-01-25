#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --gres=gpu:V100:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=/home/jrottmay/jupyter.log
#SBATCH --error=/home/jrottmay/jupyter_error.log

hostname

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser