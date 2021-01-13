#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=120:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

srun hostname
srun python3 code/3_classification.py
