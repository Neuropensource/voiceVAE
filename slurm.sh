#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="spectroVAE"
#SBATCH --time=00:00:10
#SBATCH --output="output_screen"
#SBATCH --error="output_error"
# echo " $SLURM_ARRAY_TASK_ID "
echo hello_world
conda activate voiceVAE
srun python trainingSimple.py