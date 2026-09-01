#!/bin/bash 

#SBATCH --job-name finetune
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node h100:1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --mem=32G
#SBATCH --time=72:00:00

module purge
module load anaconda3

SUBSETS=(tce-subset mace-subset)
SUBSET=${SUBSETS[$SLURM_ARRAY_TASK_ID]}
uv run mace_run_train \
    --work_dir="potential-dirs/${SUBSET}" \
    --name="MACE_${SUBSET}" \
    --foundation_model="large" \
    --multiheads_finetuning=False \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --train_file="splits/${SUBSET}.xyz" \
    --valid_file="splits/validation.xyz" \
    --test_file="splits/test.xyz" \
    --lora=True \
    --lora_rank=8 \
    --lora_alpha=2.0 \
    --energy_weight=1.0 \
    --forces_weight=1.0 \
    --E0s="estimated" \
    --lr=0.005 \
    --weight_decay=0.0 \
    --ema \
    --ema_decay=0.995 \
    --amsgrad \
    --clip_grad=10.0 \
    --max_num_epochs=50 \
    --device=cuda \
    --seed=0
