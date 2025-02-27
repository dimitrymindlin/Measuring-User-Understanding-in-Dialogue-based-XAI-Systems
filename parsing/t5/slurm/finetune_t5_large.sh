#!/bin/bash
#SBATCH --job-name=large_t5
#SBATCH --output=./parsing/t5/large_finetune_t5.txt
#SBATCH --time=21-00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s5
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=15000MB

srun python parsing/t5/start_fine_tuning.py --gin parsing/t5/gin_configs/t5-large.gin --dataset $DATASET
