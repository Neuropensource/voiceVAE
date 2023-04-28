#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="mtrf_spectrogram"
#SBATCH --time=20:00:00
#SBATCH --output="output_screen.txt"
#SBATCH --error="output_error.txt"
#echo "$SLURM_ARRAY_TASK_ID"
conda activate mne
srun mtrf_spectrogram_cluster.py