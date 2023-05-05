#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="vanillaVAE"
#SBATCH --time=24:00:00
#SBATCH --output="cluster_outputs/Out_slurm3"
#SBATCH --error="cluster_outputs/Err_slurm3"
# echo " $SLURM_ARRAY_TASK_ID "

echo hello_world
conda activate torch_cuda11_charly
srun python training_cVAE.py --device=cluster 