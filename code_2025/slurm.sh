#!/bin/bash

#SBATCH --job-name=relu
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --partition=ou_bcs_low

python main.py  --dataset cifar10 \
                --num_classes 10 \
                --hidden 60 \
                --root ./data \
                --batch_size 256 \
                --epochs 50 \
                --lr 0.5e-2 \
                --seed 0 \
                --device cuda \
