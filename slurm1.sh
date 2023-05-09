#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name="vanillaVAE"
#SBATCH --time=72:00:00
#SBATCH --output="cluster_outputs/Out_slurm1"
#SBATCH --error="cluster_outputs/Err_slurm1"
# echo " $SLURM_ARRAY_TASK_ID "

echo hello_world
conda activate torch_cuda11_charly
srun python training_cVAE.py --device=cluster

#--exclude=lifnode1,asfalda1,sensei1,lisnode3,diflives1,see4c1