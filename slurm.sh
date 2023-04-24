#!/bin/bash
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="spectroVAE"
#SBATCH --time=3:00:00
#SBATCH --output="output_screen"
#SBATCH --error="output_error"
# echo " $SLURM_ARRAY_TASK_ID "
conda activate charly
srun python trainingSimple.py 