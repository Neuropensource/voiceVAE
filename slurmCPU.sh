#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="spectroVAE"
#SBATCH --time=24:00:00
#SBATCH --output="cluster_outputs/screen"
#SBATCH --error="cluster_outputs/error"
# echo " $SLURM_ARRAY_TASK_ID "

echo hello_world
conda activate voiceVAE
srun python trainingSimple.py --device=cluster