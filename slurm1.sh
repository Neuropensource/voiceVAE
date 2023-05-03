#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --job-name="cVAE"
#SBATCH --time=24:00:00
#SBATCH --output="cluster_outputs/Out_slurm1"
#SBATCH --error="cluster_outputs/Err_slurm1"
# echo " $SLURM_ARRAY_TASK_ID "

echo hello_world
conda activate voiceVAE
srun python main.py --device=cluster

#SBATCH --exclude=lifnode1,asfalda1,sensei1,lisnode3,diflives1,see4c1