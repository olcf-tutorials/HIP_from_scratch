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
16384x16384 matrix and each element of the matrix is 1. In memory, we represent this
matrix as a 1 dimensional array. We have two kernels, one that calculates the sum of the
rows, and one that calculates the sum of the columns. We also allocate an array of size
16384 in the main memory and GPU memory to store the row sum or the column sum. We set the
block size to 256 and the grid size (i.e. number of blocks) such that there are at least
16384 threads, so one thread for each row or column we are summing. Much of
the HIP calls should now look familiar to you after the seeing [the earlier module with
the vector add example](TODO). Briefly, let us go over the `row_sums` and `column_sums`
kernel .

```
// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds) {

  // create typical 1D thread index from built-in variables
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < ds) {
    float sum = 0.0f;

    // write a for loop that will cause the thread to iterate across
    // a row, keeping a running sum, and write the result to sums
    for (size_t i = 0; i < ds; i++)
      sum += A[idx * ds + i];
    sums[idx] = sum;
  }
}
```

The row sum is done by assigning each thread up to 16384 (i.e. the size of the matrix) one
of the rows of the matrix. Each thread will then sum the values in that row. The `idx <
ds` is to ensure that if there are any more threads than there are rows, the extra threads
just no-op (ideally you would make sure that no thread is idle and avoid thread divergence
as much as possible. In our case, we are actually starting 16384 threads because of how we
calculate the grid size for our kernel launch, so there are no extraneous threads in this
example. The if statement is just a precaution). Since the matrix is represented as a 1
dimensional array, each thread needs to jump to the starting point of the row
corresponding to its `idx` value and then sum the values in that row.

```
// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds) {
  // create typical 1D thread index from built-in variables

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < ds) {
    float sum = 0.0f;

    // write a for loop that will cause the thread to
    // iterate down a column, keeeping a running sum,
    // and write the result to sums
    for (size_t i = 0; i < ds; i++)
      sum += A[idx + ds * i];
    sums[idx] = sum;
  }
}
```

The column sum operates on the same principle as the row sum: each thread is assigned a
column. Each thread will step to the next value in the column, which is `ds` items away
in the 1D array from the previous value in the column. 


#### Building and running the code

Make sure you have access to a system with HIP installed with a ROCm or CUDA backend. Make
sure you update the submit scripts with the project in the batch job directives and in the
`OUTPUT` variable.

For Summit, run the following commands
```
module load cuda/11.4.0
module load hip-cuda/5.1.0
hipcc -o matrix_sums_unoptimized matrix_sums_unoptimized.cpp

# submit job
bsub submit_summit_unoptimized.lsf
```

For Spock/Crusher
```
module load rocm/5.1.0
hipcc -o matrix_sums_unoptimized matrix_sums_unoptimized.cpp

# submit job
sbatch submit_frontier_unoptimized.sbatch
```

The submit scripts for Summit and Crusher run the executable with the nsys profiler and
the rocprof profiler respectively.

#### Examining the Results of the Kernel Profiling

From the output file of the Summit run, you will see a section titled `CUDA Kernel
Statistics` with the table summarizing the run time of
the two kernels. It will look something like this

```
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)                         Name
 -------  ---------------  ---------  ------------  ------------  ------------  -----------  --------------------------------------------------
    60.0          3263784          1     3263784.0       3263784       3263784          0.0  row_sums(const float *, float *, unsigned long)
    40.0          2176528          1     2176528.0       2176528       2176528          0.0  column_sums(const float *, float *, unsigned long)
```

You will notice that the `row_sums` kernel is actually slower than the `column_sums`
kernel. Why is that?

This is because the `column_sums` kernel actually makes better use of _memory
coalescing_. Recall that when you read a piece of data from memory, it will load that
memory into the cache along with some data that was adjacent to it because there is a
reasonable assumption that if you need some data you will likely also need the data next
to it. We can take advantage of this by making sure that threads that are adjacent to each
other (i.e. threads in the same block or same wavefront even) make use data that is
adjacent to each other. So for `column_sums`, when thread with `idx 0` accesses A[0], `idx
1` accesses A[1], `idx 2` accesses A[2] and so on. Since A[1], A[2], A[3] are all adjacent
to each other, they will all brought to the cache together, and we don't have to do
separate memory lookups for getting the data for `idx 2` and `idx 3` which are executing
in the same wavefront. This saves a lot of time.

Contrast this with `row_sums`. When `idx 0` accesses A[0], `idx 1` is accessing `A[1*ds ==
16384]`, `idx 2` is accessing `A[2*ds == 32768]` and so on. Since `idx` 0,1,2 are all in
the same wavefront but are accessing data that is far away from each other, the data is
not all in the cache and each thread will have to wait for a cache miss to happen, the
data to be copied from memory into the cache, and then do its operation. And since threads
in a wavefront execute in lockstep, each time a memory access happens the whole wavefront
is held up because each thread in the wavefront is effectively being serviced one at a
time rather than all at once. So you can see how this can slow things down a lot. There is
no _memory coalescing_ here.

(TODO: draw a diagram to accompany the above explanation).

(TODO: when rocprof timing information is fixed, add a section covering the rocprof output
as well).


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

