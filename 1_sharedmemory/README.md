Tutorial for shared memory features for GPU

## Memory Coalescing
(TODO: the developer.amd matrix transpose example isn't really a good example of memory
coalescing as it looks like the number of write and read transactions is more or less the
same between the naive and lds examples.)

Effective use of cache is something to keep in mind while doing GPU operations. When a
memory access to data in the GPU memory occurs, a block of data is actually brought into
the cache in the CU (TODO: citation needed - is cache in the CU). If different threads in
a block are accessing wildly different parts of the memory, you will be facing a lot of
cache misses as the memory operations have to be serialized, the cache cleared out and new
data is brought into the cache to fulfill each thread's request. So it is good practice to
structure your kernel code such that adjacent threads in a block access adjacent data to
maximize use of the cache. This way of organizing your data access to take advantage of
the cache and reduce the number of memory transactions is called _memory coalescing_.

### How Effective is Memory Coalescing

For example, let us take the `matrix_sums_unoptimized.cpp` code. Here we are creating a
16384x16384 matrix and each element of the matrix is 1. We have two kernels, one that
calculates the sum of the rows, and one that calculates the sum of the columns. Much of
the HIP calls should now look familiar to you after the seeing [the earlier module with the
vector add example](TODO). Briefly, let us go over the `row_sums` and `column_sums` kernel
. 


```

(TODO: Try creating an example where you are summing the
rows vs summing the columns of a matrix to show effect of memory coalescing)
(TODO: show instructions on how to run the nvidia profiler on this for summit and the
rocprof on crusher and show the performance differences).

# GPU Shared Memory with HIP

So how do we improve the performance of the sum of the rows. We can take advantage of GPU
shared memory to bring the data even closer to the threads (TODO: is the LDS on the
CUs?). This is called Local Data Share or LDS. Let us look at a modified example of the row
sum kernel in (TODO: filename). Whereas column sum is taking advantage of the cache lines
(TODO: explain how cache lines are set up on the GPU). 

(TODO: show how to run profiler on Summit and crusher and compare results with previous row sum run that didn't use
shared memory)



## Memory banks and bank conflicts
TODO: clean up and add accurate information.
TODO: we can use the matrix transpose example here as we found it is a good example to use 
to teach about bank conflicts and how to move data to minimize the conflicts.

The shared memory that is visible to all the threads of a block are made up of memory
banks. Each memory bank is of TODO bit words. If multiple threads in the same wavefront
from a block try to access the same bank (even if they are trying to use different data),
this is a bank conflict. These accesses have to be serialized which could affect the
latency. A way to mitigate bank conflicts is to try and make sure that the threads are set
up in such a way that threads in a wavefront are accessing different memory banks to avoid
the bank conflicts. 


For example, let us look at code that does a matrix transpose, utilizing the LDS. Each
thread in a block first identifies a value in the in matrix and moves it to the LDS. All
the threads are then synced and then the values from the LDS are moved to the appropriate
locations in the out matrix. (TODO: supposedly this was supposed to be faster and have
fewer write memory transactions than the naive way where we copy directly from the in
matrix to the outmatrix without using the LDS intermediary, but rocprof doesn't show that
even though their video tutorial does. I'm not sure what's going on here. However the
matrix transpose is a good example for bank conflicts because copying from  `in matrix ->
lds[x][y] -> out matrix` has a lot of bank conflicts, but copying `in matrix -> lds[y][x]
-> out matrix` has 0 bank conflicts because each thread in a warp is accessing a different
memory bank because it is accessing columnwise (whereas in the previous each thread in the
warp access data row wise which may all fall on the same memory bank leading to
conflicts). (TODO: have someone verify the previous statement if it's true or not).

(TODO: show how to use profiler on summit and crusher to see the speed results and print 
out the bank conflict information)

## Building and running the example with a profiler


TODOs:
1. What is the maximum shared memory in the AMD GPUs?
2. How are memory banks set up? how many memory banks constitute shared memory? how much
   data per memory bank? How do you handle bank conflicts better?
3. What is memory coalescing? How can using shared memory help?
4. Why do you want to use shared memory?
5. Why do you use __syncthreads? (to avoid race conditions between threads in the same block accessing the shared memory).

