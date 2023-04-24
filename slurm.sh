# !/bin/bash
# SBATCH --partition=partition
# SBATCH --nodes=Nnodes
# SBATCH --ntasks-per-node=Ntasks
# SBATCH --gres=gpu:Ngpus
# SBATCH --cpus-per-task=Ncpus
# SBATCH --job-name="JobName"
# SBATCH --time=TIME
# SBATCH --output="output_screen"
# SBATCH --error="output_error"
# echo " $SLURM_ARRAY_TASK_ID "
conda activate charly
srun python file.py 