#!/bin/bash

#SBATCH -A PROJ123
#SBATCH -J matrix_sums_unoptimized
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 00:05:00

module load rocm/5.1.0

srun -N1 -n1 -c1 --gpus-per-task=1 rocprof --stats -i metrics_matrix_sums_unoptimized_kernels.txt -o metrix_matrix_sums_unoptimized.csv ./matrix_sums_unoptimized