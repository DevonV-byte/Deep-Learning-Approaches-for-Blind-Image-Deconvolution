#!/bin/bash -l
#SBATCH --cluster=genius
#SBATCH --account=lp_h_cs_master
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"
#SBATCH --partition=gpu_p100

# nvidia-smi

source /data/leuven/303/vsc30380/slurmhooks

# Load Python module and activate environment
module load Python/3.11.3-GCCcore-12.3.0
source PyPipe/bin/activate
pip install --upgrade pip

# Set TensorFlow log level to suppress warnings
export TF_CPP_MIN_LOG_LEVEL=2

# run convolution script
time python control.py --dataset-path ../Base_Images/landmark/ --sample-percentage 0.01 --batches 10 --num-kernels 10 --augmentation-rate 1 --grayscale --epochs 3 --learning-rate 0.0001 --loss-function mse --conv-mode --images-per-kernel 5