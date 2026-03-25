#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=lp_h_cs_master
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"
#SBATCH --partition=gpu_h100

# WICE Test

# nvidia-smi

# Load CUDA module
module load CUDA/12.1.1

# Activate Conda environment
source /data/leuven/303/vsc30380/slurmhooks
conda activate tf_env

# Set TensorFlow log level to suppress warnings
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/leuven/common/software/CUDA/12.1.1
export TF_CPP_MIN_LOG_LEVEL=2

# run convolution script
time python control.py --dataset-path ../Base_Images/landmark/ --sample-percentage 0.3 --batches 261 --augmentation-rate 1 --test