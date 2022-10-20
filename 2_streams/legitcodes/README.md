# Streams and Events

## Streams
Thus far, we have seen only the execution of one kernel launch or one `Host<->Device`
memory copy at a time. If the kernel is not saturating the GPU with the work it is doing,
and if your code has work that could be overlapped, you could take advantage of **HIP
Streams**. 

The idea here is that you can create different pipes or _streams_ where to which you can
assign different tasks that don't depend on each other and thus could execute at the same
time and make better use of the GPU. This could get you some additional throughput.  Let
us look at a couple of code examples to make this concrete.

In `single_stream.cpp`, we are allocating host and GPU memory for two arrays A and B that
each hold 1024^2 32x32 matrices and filling it with random values.  And then in a loop in
each iteration we copy the data from A and B corresponding to one 32x32 matrix to their
respective `d_A` and `d_B` locations in GPU memory, multiply the matrices using the
`matrix_multiply` kernel, and copy the result back to host memory to the appropriate
location the array C. We do a warmup run before the loop as it is necessary to get the GPU
to the right frequency (TODO: fact check) without which the timing information will be
skewed due to the excess time taken by whichever operation takes place first on the GPU.
You will notice that the matrices are allocated to host memory using `HipHostMalloc`
instead of regular `malloc`. We will cover this in the Pinned Memory section. You will
also notice that on host memory we are allocating memory to a single pointer for A and B,
when it might have been easier to understand and iterate through if A and B were each an
array of pointers with one pointer for each matrix. The reason for this will also be
covered in the Pinned Memory section. For now, take a moment to understand the code as it
is and how it works.

```
// The copy matrices, run kernel, copy result loop
  for (int m = 0; m < num_matrices; m++) {
    gpuErrorCheck(hipMemcpy(&d_A[m * N * N], &A[m * N * N],
                            N * N * sizeof(double), hipMemcpyHostToDevice));
    gpuErrorCheck(hipMemcpy(&d_B[m * N * N], &B[m * N * N],
                            N * N * sizeof(double), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0, 0,
                       &d_A[m * N * N], &d_B[m * N * N], &d_C[m * N * N]);

    gpuErrorCheck(hipMemcpy(&C[m * N * N], &d_C[m * N * N],
                            N * N * sizeof(double), hipMemcpyDeviceToHost));

  }


```

We also use `hipEvent` calls to mark the before and after of the loop in order to measure 
the time taken by the loop.

By default, all GPU code run in the same stream i.e. stream 0 which is the default stream. (TODO: fact check
this). Any GPU operation in a stream must wait for the previous operation in
that same stream to finish before it can execute. In `single_stream.cpp` when you're
multtiplying so many different matrices, you will find that the sequential nature
of these operations can be a bottleneck. You might have other code that might be in a 
similar situation. Splitting the work across multiple streams can alleviate this bottleneck.
Also, `hipMemcpy` is a blocking operation i.e. the host thread will block till the copy
operation finishes. It might be useful to have a copy operation
that is asynchronous.

Let's take a look at the loop in `multiple_streams.cpp`. It is identical to
`single_stream.cpp` with a few exceptions. A variable `num_streams=8` sets the number of
streams we will use. In the code snippet below we define a `hipStream_t` array and
populating it with the hipStream objects with `hipStreamCreate`. In the loop, as before,
we are trying to multiply the matrices at the same index in the A and B matrix array and
copying the result in the the C matrix array. Unlike in the `single_stream.cpp` example,
we are using the `hipMemcpyAsync` to copy the data between the host and GPU, and we are
passing it a stream object from the array of stream objects we defined earlier. The
`hipMemcpyAsync` is a non blocking operation and returns immediately. It queues up the
memcpy operation in the stream that it was given and continues to the next operation. (In
HIP, you can think of any non blocking call like `hipMemcpyAsync` or `hipLaunchKernelGGL`
as if they are queueing up that particular operation in the specified stream, and the
scheduler in the GPU eventually executes it). If
you'll remember that the `hipLaunchKernelGGL` is also a non blocking operation because it
is doing a similar thing of queueing up the operation in the given stream. It's just that
this time we are specifying a particular stream instead of it automatically being run
in the the default stream 0. There
is an obvious increase in throughput this way by setting IO operations and corresponding
computations on the data in different streams.


(TODO: is there an upper limit on the number of streams? What are the diminishing returns for
increasing number of streams? What's a good heuristic for how many streams you need?)

```
  hipStream_t streams[num_streams];

  for (int i = 0; i<num_streams; i++) {
    gpuErrorCheck(hipStreamCreate(&streams[i]));
  }
...
...
// The copy matrices, run kernel, copy result loop
  for (int m = 0; m < num_matrices; m++) {
    int stream_number = m % num_streams;
    gpuErrorCheck(
        hipMemcpyAsync(&d_A[m*N*N], &A[m*N*N], N * N * sizeof(double), hipMemcpyHostToDevice,streams[stream_number] ));
    gpuErrorCheck(
        hipMemcpyAsync(&d_B[m*N*N], &B[m*N*N], N * N * sizeof(double), hipMemcpyHostToDevice, streams[stream_number]));
  
  hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0, streams[stream_number], &d_A[m*N*N], &d_B[m*N*N],  &d_C[m*N*N]);
    gpuErrorCheck(
      hipMemcpyAsync(&C[m*N*N], &d_C[m*N*N], N * N * sizeof(double), hipMemcpyDeviceToHost, streams[stream_number]));
  }
```

(Note: Here we're performing several small copies from host to device and back one matrix
at a time in the loop simply to demonstrate streams for both the async memcpy and kernel
launch operations and their performance benefits.  Performing one big copy of the whole
array of matrices outside the loop instead of many smaller copies within the loop in this
particular example is actually more efficient (try it out and see! or look at
`single_stream_onebigcopy.cpp` and `multiple_streams_onebigcopy.cpp`). The needs of your
particular application may vary.

Let us now try to compile and run these. On Crusher:

```
# to compile
module load rocm/5.2.0
hipcc -o single_stream single_stream.cpp
hipcc -o multiple_streams multiple_streams.cpp

# submit job
sbatch submit_single_stream_multiple_streams.sbatch
```

The submitted job will execute both `single_stream` and `multiple_streams` and output the
time taken by loops. The output might look something like this.

```
running single_stream
single stream 46248.621094 milliseconds hipEvent
single stream 46248.642000 milliseconds cpu
__SUCCESS__
running multiple_streams
multiple stream 33903.328125 milliseconds hipEvent
multiple stream 33903.352000 milliseconds cpu
__SUCCESS__
```
 
You can see that the `multiple_streams` takes less time due to the concurrency we get
from splitting work into different streams. We're measuring the time
with both `hipEvent` and also with `time.h` in the code just to verify if the timing
 is measured correctly.

A thing to note that your kernel might saturate the whole GPU and may not necessarily
benefit from splitting up the work into different streams. As with all things, do your
own experiments to determine which course of action works best for your application.


## Pinned Memory

Recall that in our code, we are allocating the memory for matrices A, B, and C on the host
memory with `hipHostMalloc` instead of plain `malloc`. `hipHostMalloc` allocates __pinned
memory__, which is memory that isn't paged in and out. When we allocate memory with
malloc, we are actually allocating _pageable memory_. This means that when accessing that
data in this location may not necessarily be resident in RAM, but might be in swap. So
there is a possibility of page fault when we try to read from this location which can cost
time. So any data access in this memory location must go through the CPU, GPU cannot
access pageable memory directly. Pinned memory is page locked, which means that the data
written to this memory will not be paged out. So the underlying memory access code can
bypass the CPU mechanism of accessing memory and instead access it directly from the RAM
which is more efficient (TODO: fact check).  Also, when we copy data from memory allocated
with malloc to the GPU memory, it gets staged through temporary pinned memory that is
created in the background that is invisible to the developer, which also costs time. Using
pinned memory directly saves us time. Data transfers from pinned memory is a lot faster
than pageable memory, as seen in `single_stream_bandwidthtest.cpp`. Build and run it just
like the previous example, replacing the executable name in the batch script
appropriately.  

Something to keep in mind is that since the pinned memory is locked and cannot be paged
out, this reduces the available physical memory for the rest of your application until
memory is freed.  You will have to make the calculation of how much pinned memory to use
depending on your application.

You will also remember that we are also performing one single large allocation for the
entire array matrices with `hipHostMalloc` instead of many small allocations with one
allocation for each matrix. If we did one allocation with `hipHostMalloc` for each matrix,
we will run into the `hipErrorOutOfMemory` error. But it's
not because we ran out of physical memory on the host (we are allocating the same amount
of memory as before, just doing more individual allocations) but because we run into the mmap
limit (which you can check with `sysctl vm.max_map_count`). So keep that in mind if you
get a `hipErrorOutOfMemory` when using `hipHostMalloc` but you don't think you've actually
used all the memory, you might be hitting that limit so might want to consider adjusting
how you do your memory allocation.


## Events

We've been using events so far to measure the latency of the GPU operations using
`hipEventRecord`. Another use of events is to synchronize the operations of different
streams. Say you had two streams `stream1` and `stream2`. `stream1` is used to copy the
data into GPU memory, while `stream2` is used to execute the kernels on the data moved by
`stream1` and copy the result back into host memory (Note: I don't know if there is an
actual use case where someone might want to do something like this. I'm just using it as
an example to demonstrate stream synchronization).  In this situation, we would want to make sure
that the kernel being launched in `stream2` doesn't try to operate on data that hasn't
been copied yet to GPU memory by `stream1`. One way to do this is seen in the
`hipeventsynchronize_example.cpp` which we modify from our earlier `multiple_streams.cpp`. The
modified (copy matrices, run kernel, copy result) loop is shown below.

```cpp
  // creating two streams and the event the streams will use to synchronize with

  hipStream_t stream1;
  hipStream_t stream2;

  gpuErrorCheck(hipStreamCreate(&stream1));
  gpuErrorCheck(hipStreamCreate(&stream2));

  hipEvent_t datatransfer;
  gpuErrorCheck(hipEventCreate(&datatransfer));

  // The copy matrices, run kernel, copy result loop
  for (int m = 0; m < num_matrices; m++) {
    gpuErrorCheck(hipMemcpyAsync(&d_A[m * N * N], &A[m * N * N],
                                 N * N * sizeof(double), hipMemcpyHostToDevice,
                                 stream1));
    gpuErrorCheck(hipMemcpyAsync(&d_B[m * N * N], &B[m * N * N],
                                 N * N * sizeof(double), hipMemcpyHostToDevice,
                                 stream1));

    // This will insert the datatransfer event in stream1 after the above
    // hipMemcpy operations
    gpuErrorCheck(hipEventRecord(datatransfer, stream1));

    // you could do some additional operations here if you want to, for example
    // line up more operations on stream1.
    // ...

    // This will block till all the operations on stream1 (up until the point
    // where we had called hipEventRecord on the datatransfer event) is
    // completed
    gpuErrorCheck(hipEventSynchronize(datatransfer));

    // The above method is useful in a situation where you want to mark a point
    // in a stream and then continue to line up operations in the stream, and
    // then at some point later in your code you call hipEventSynchronize to
    // block on the host thread and wait till all the GPU operations in the
    // stream up until that marked event is completed (and if you don't necessarily
    // want to block till all the GPU operations so far are completed, just the
    // ones till the marked event point). hipEventRecord and
    // hipEventSynchronize do not have have to be called back-to-back. If you
    // want to block and synchronize the stream immediately, you can simply
    // just call hipStreamSynchronize
    // gpuErrorCheck(hipStreamSynchronize(stream1));

    // we don't have to do any synchronization calls on stream2 since stream1 doesn't
    // depend on on anything needing to be completed in stream2. But the
    // operations in stream2 do depend on the Memcpy operations in stream1
    // completing correctly before any operations can be done on that data.
    hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0,
                       stream2, &d_A[m * N * N], &d_B[m * N * N],
                       &d_C[m * N * N]);

    gpuErrorCheck(hipMemcpyAsync(&C[m * N * N], &d_C[m * N * N],
                                 N * N * sizeof(double), hipMemcpyDeviceToHost,
                                 stream2));
  }

  // blocks on the host thread till all the GPU operations on all streams is
  // completed
  gpuErrorCheck(hipDeviceSynchronize());
```

A thing to note also is that the `hipDeviceSynchronize` call will block on the host thread
till all the operations on the GPU on all streams up until that point are completed.


Another feature that is available to synchronize streams is the `hipStreamWaitEvent` call
if you want an asynchronous non blocking call (unlike the `*Synchronize` calls which block
the host thread at the point they are called). You will mark a point in a stream (say
`stream1`) with an event using `hipEventRecord` but then you can make another stream (say
`stream2`) wait till all the GPU operations in `stream1` till that marked event is
complete by calling `hipStreamWaitEvent(stream2, datatransfer, 0))` without forcing the
host thread to also block. See the snippet from
the `hipstreamwaitevent_example.cpp` below.

```cpp
  // creating two streams and the event that we'll synchronize with

  hipStream_t stream1;
  hipStream_t stream2;

  gpuErrorCheck(hipStreamCreate(&stream1));
  gpuErrorCheck(hipStreamCreate(&stream2));

  hipEvent_t datatransfer;
  gpuErrorCheck(hipEventCreate(&datatransfer));

  // The copy matrices, run kernel, copy result loop
  for (int m = 0; m < num_matrices; m++) {
    gpuErrorCheck(hipMemcpyAsync(&d_A[m * N * N], &A[m * N * N],
                                 N * N * sizeof(double), hipMemcpyHostToDevice,
                                 stream1));
    gpuErrorCheck(hipMemcpyAsync(&d_B[m * N * N], &B[m * N * N],
                                 N * N * sizeof(double), hipMemcpyHostToDevice,
                                 stream1));

    // This will insert the datatransfer event in stream1 after the above
    // hipMemcpy operations
    gpuErrorCheck(hipEventRecord(datatransfer, stream1));

    // The hipStreamWaitEvent call is non blocking and will return immediately,
    // unlike the the hipEventSynchronize call which will block on the host
    // thread till all the GPU operations in the stream1 until the
    // hipEventRecord are done. Also note that it is actually stream2 that will
    // wait on the datatransfer event (which marks a point in stream1). So stream2
    // synchronizes with stream1 just like it is synchronized in the loop in
    // hipeventsynchronize_example.cpp, just without blocking on the host thread.
    gpuErrorCheck(hipStreamWaitEvent(stream2, datatransfer, 0));

    // Since hipStreamWaitEvent is non blocking, the code will continue
    // executing. But the GPU operations we place in stream2 are just lined up
    // in the stream2 and don't actually begin executing till the datatransfer
    // event at the point marked by hipEventRecord in stream1 is reached.
    hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0,
                       stream2, &d_A[m * N * N], &d_B[m * N * N],
                       &d_C[m * N * N]);

    gpuErrorCheck(hipMemcpyAsync(&C[m * N * N], &d_C[m * N * N],
                                 N * N * sizeof(double), hipMemcpyDeviceToHost,
                                 stream2));
  }

  gpuErrorCheck(hipDeviceSynchronize());

```


## Resources:
GPU Programming Concepts (Part 2) (covers streams): https://www.youtube.com/watch?v=i0GzebZKi10

## Exercises:
1. Play around with setting different values for `num_streams` in `multiple_streams.cpp`
to see if it affects the timing.
