#!/bin/bash
#SBATCH -A stf007uanofn
#SBATCH -J singstream_multstream
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 00:15:00

echo "running single_stream"
srun -N1 -n1 -c1 --gpus-per-task=1 ./single_stream
echo "running multiple_streams"
srun -N1 -n1 -c1 --gpus-per-task=1 ./multiple_streams
