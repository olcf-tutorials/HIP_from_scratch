#!/bin/bash

#SBATCH -A PROJ123
#SBATCH -J add_vec_hip
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 00:05:00

srun -N1 -n1 -c1 --gpus-per-task=1 ./vadd_hip