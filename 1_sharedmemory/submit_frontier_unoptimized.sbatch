#!/bin/bash

#SBATCH -A STF007uanofn
#SBATCH -J matrix_sums_unoptimized
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 00:05:00

module load rocm/5.1.0

OUTPUT=/gpfs/alpine/stf007/scratch/${USER}
# bug with -i flag means this won't produce correct timing output
#srun -N1 -n1 -c1 --gpus-per-task=1 rocprof --stats -i metrics_matrix_sums_kernels.txt -o $OUTPUT/metrics_matrix_sums_unoptimized.csv ./matrix_sums_unoptimized

srun -N1 -n1 -c1 --gpus-per-task=1 rocprof --stats  -o $OUTPUT/metrics_matrix_sums_unoptimized.csv ./matrix_sums_unoptimized

cp $OUTPUT/metrics_matrix_sums_unoptimized.*csv $SLURM_SUBMIT_DIR
