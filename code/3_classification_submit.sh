#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=180:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G

srun hostname
srun python3 ../code/3_classification.py
