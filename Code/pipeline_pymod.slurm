#!/bin/bash -l
#SBATCH --cluster=genius
#SBATCH --account=lp_h_cs_master
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:45:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"
#SBATCH --partition=gpu_p100

nvidia-smi

source /data/leuven/303/vsc30380/slurmhooks

# Load Python module and activate environment
module load Python/3.11.3-GCCcore-12.3.0
source PyPipe/bin/activate

# Set TensorFlow log level to suppress warnings
export TF_CPP_MIN_LOG_LEVEL=2

# run convolution script
time python control.py --dataset-path ../Base_Images/landmark/ --sample-percentage 0.3 --batches 261 --augmentation-rate 1