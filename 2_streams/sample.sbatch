#!/bin/bash

#SBATCH -A stf007uanofn
#SBATCH -J test
#SBATCH -N 1
#SBATCH -t 01:00:10

echo "hello my name is subil"

#for i in {1..10}
#do
#srun -N1 -n1 -G1 ./multiple_streams_alt_pinnedmem_matmul | awk '{print $3" "$5}' | tee -a multstreamtimings 
#srun -N1 -n1 -G1 ./single_stream_alt_pinnedmem_matmul | awk '{print $3" "$5}' | tee -a singlestreamtimings 
#done


#for i in {1..10}
#do
#srun -N1 -n1 -G1 ./multiple_streams | awk '{print $3" "$5}' | tee -a timings/multiple_streams_timings
#srun -N1 -n1 -G1 ./single_stream | awk '{print $3" "$5}' | tee -a timings/single_stream_timings 
#done

#for i in {1..10}
#do
#srun -N1 -n1 -G1 ./multiple_streams_alt | awk '{print $3" "$5}' | tee -a timings/multiple_streams_alt_timings
#srun -N1 -n1 -G1 ./single_stream_alt | awk '{print $3" "$5}' | tee -a timings/single_stream_alt_timings 
#done

#for i in {1..10}
#do
#srun -N1 -n1 -G1 ./multiple_streams_onebigcopy | awk '{print $3" "$5}' | tee -a timings/multiple_streams_onebigcopy_timings
#srun -N1 -n1 -G1 ./single_stream_onebigcopy | awk '{print $3" "$5}' | tee -a timings/single_stream_onebigcopy_timings 
#done

for i in {1..10}
do
srun -N1 -n1 -G1 ./multiple_streams_dgemm | awk '{print $3" "$5}' | tee -a timings/multiple_streams_dgemm_timings
srun -N1 -n1 -G1 ./single_stream_dgemm | awk '{print $3" "$5}' | tee -a timings/single_stream_dgemm_timings 
done

#for i in {1..10}
#do
#srun -N1 -n1 -G1 ./multiple_streams_matmul | awk '{print $3" "$5}' | tee -a timings/multiple_streams_matmul_timings
#srun -N1 -n1 -G1 ./single_stream_matmul | awk '{print $3" "$5}' | tee -a timings/single_stream_matmul_timings 
#done
