Tutorial for shared memory features for GPU

# GPU Shared Memory with HIP

Often, when your operating on data, there may be cases where a thread is operating on one
piece of data and then has to operate on a different piece of data from a different
thread. Doing this naively would require writing to GPU memory and then reading it back,
which can be slow. AMD GPUs provide shared memory at the block level that would greatly
help decrease the latency. This is called Local Data Share (LDS). Reads and writes on LDS
are 100x faster compared to global memory. 


## Memory Coalescing
(TODO: the developer.amd matrix transpose example isn't really a good example of memory
coalescing as it looks like the number of write and read transactions is more or less the
same between the naive and lds examples. Try creating an example where you are summing the
rows vs summing the columns of a matrix to show effect of memory coalescing)

## Memory banks and bank conflicts

The shared memory that is visible to all the threads of a block are made up of memory
banks. Each memory bank is of TODO bit words. If multiple threads in the same wavefront
from a block try to access the same bank (even if they are trying to use different data),
this is a bank conflict. These accesses have to be serialized which could affect the
latency. A way to mitigate bank conflicts is to try and make sure that the threads are set
up in such a way that threads in a wavefront are accessing different memory banks to avoid
the bank conflicts. 


TODOs:
1. What is the maximum shared memory in the AMD GPUs?
2. How are memory banks set up? How do you handle bank conflicts better?
3. What is memory coalescing? How can using shared memory help?
4. Why do you want to use shared memory?
5. Why do you use __syncthreads? (to avoid race conditions between threads in the same block accessing the shared memory).

