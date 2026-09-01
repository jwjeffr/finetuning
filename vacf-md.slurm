#!/bin/bash 

#SBATCH --job-name vdos
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node h100:1
#SBATCH --cpus-per-task=5
#SBATCH --array=0-5
#SBATCH --mem=32GB
#SBATCH --time=72:00:00

module purge
module load anaconda3

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

SYSTEMS=(WSTe WSeTe MoSTe MoSeTe TaSTe TaSeTe)
python "$SLURM_SUBMIT_DIR/vacf-md.py" "${SYSTEMS[$SLURM_ARRAY_TASK_ID]}" "$@"
