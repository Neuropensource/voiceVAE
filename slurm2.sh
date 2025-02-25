#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="spectroVAE"
#SBATCH --time=24:00:00
#SBATCH --output="cluster_outputs/Out_slurm2"
#SBATCH --error="cluster_outputs/Err_slurm2"
# echo " $SLURM_ARRAY_TASK_ID "

echo hello_world
conda activate voiceVAE
srun python trainingSimple.py --device=cluster --partial=T