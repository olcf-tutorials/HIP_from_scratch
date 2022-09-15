#include "hip/hip_runtime.h"
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 *
 *
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel is used to
 * initialize an array to a specific value, after which the array is
 * copied to the host (CPU) memory.  To increase performance, multiple
 * kernel/memcopy pairs are launched asynchronously, each pair in its
 * own stream.  Devices with Compute Capability 1.1 can overlap a kernel
 * and a memcopy as long as they are issued in different streams.  Kernels
 * are serialized.  Thus, if n pairs are launched, streamed approach
 * can reduce the memcopy cost to the (1/n)th of a single copy of the entire
 * data set.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5
 * microsecond precision.
 *
 * Elapsed times are averaged over nreps repetitions (10 by default).
 *
*/

// Macro for checking errors in CUDA API calls
#define gpuErrorCheck(call)                                                    \
  do {                                                                         \
    hipError_t gpuErr = call;                                                  \
    if (hipSuccess != gpuErr) {                                                \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             hipGetErrorString(gpuErr));                                       \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

const char *sSDKsample = "simpleStreams";


const char *sDeviceSyncMethod[] =
{
    "hipDeviceScheduleAuto",
    "hipDeviceScheduleSpin",
    "hipDeviceScheduleYield",
    "INVALID",
    "hipDeviceScheduleBlockingSync",
    NULL
};

// System includes
#include <stdio.h>
#include <assert.h>

#include <hip/hip_runtime.h>


#include <sys/mman.h> // for mmap() / munmap()



__global__ void init_array(int *g_data, int *factor, int num_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<num_iterations; i++)
    {
        g_data[idx] += *factor;    // non-coalesced on purpose, to burn time
    }
}

bool correct_data(int *a, const int n, const int c)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != c)
        {
            printf("%d: %d %d\n", i, a[i], c);
            return false;
        }
    }

    return true;
}

inline void
AllocateHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes)
{
        printf("> hipHostMalloc() allocating %4.2f Mbytes of system memory\n", (float)nbytes/1048576.0f);
        // allocate host memory (pinned is required for achieve asynchronicity)
        gpuErrorCheck(hipHostMalloc((void **)pp_a, nbytes));
        *ppAligned_a = *pp_a;
}

inline void
FreeHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes)
{
        hipHostFree(*pp_a);
}


#define DEFAULT_PINNED_GENERIC_MEMORY true

int main(int argc, char **argv)
{
    int cuda_device = 0;
    int nstreams = 4;               // number of streams for CUDA calls
    int nreps = 10;                 // number of times each experiment is repeated
    int n = 16 * 1024 * 1024;       // number of ints in the data set
    int nbytes = n * sizeof(int);   // number of data bytes
    dim3 threads, blocks;           // kernel launch configuration
    float elapsed_time, time_memcpy, time_kernel;   // timing variables
    float scale_factor = 1.0f;


    bool bPinGenericMemory  = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
    int  device_sync_method = hipDeviceScheduleBlockingSync; // by default we use BlockingSync

    int niterations;    // number of iterations for the loop inside the kernel

    printf("[ %s ]\n\n", sSDKsample);




    niterations = 5;



    // enable use of blocking sync, to reduce CPU usage
    printf("> Using CPU/GPU Device Synchronization method (%s)\n", sDeviceSyncMethod[device_sync_method]);
    gpuErrorCheck(hipSetDeviceFlags(device_sync_method | (bPinGenericMemory ? hipDeviceMapHost : 0)));

    // allocate host memory
    int c = 5;                      // value to which the array will be initialized
    int *h_a = 0;                   // pointer to the array data in host memory
    int *hAligned_a = 0;           // pointer to the array data in host memory (aligned to MEMORY_ALIGNMENT)

    // Allocate Host memory (could be using hipHostMalloc or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

    // allocate device memory
    int *d_a = 0, *d_c = 0;             // pointers to data and init value in the device memory
    gpuErrorCheck(hipMalloc((void **)&d_a, nbytes));
    gpuErrorCheck(hipMemset(d_a, 0x0, nbytes));
    gpuErrorCheck(hipMalloc((void **)&d_c, sizeof(int)));
    gpuErrorCheck(hipMemcpy(d_c, &c, sizeof(int), hipMemcpyHostToDevice));

    printf("\nStarting Test\n");

    // allocate and initialize an array of stream handles
    hipStream_t *streams = (hipStream_t *) malloc(nstreams * sizeof(hipStream_t));

    for (int i = 0; i < nstreams; i++)
    {
        gpuErrorCheck(hipStreamCreate(&(streams[i])));
    }

    // create CUDA event handles
    // use blocking sync
    hipEvent_t start_event, stop_event;
    int eventflags = ((device_sync_method == hipDeviceScheduleBlockingSync) ? hipEventBlockingSync: hipEventDefault);

    gpuErrorCheck(hipEventCreateWithFlags(&start_event, eventflags));
    gpuErrorCheck(hipEventCreateWithFlags(&stop_event, eventflags));

    // time memcopy from device
    gpuErrorCheck(hipEventRecord(start_event, 0));     // record in stream-0, to ensure that all previous CUDA calls have completed
    gpuErrorCheck(hipMemcpyAsync(hAligned_a, d_a, nbytes, hipMemcpyDeviceToHost, streams[0]));
    gpuErrorCheck(hipEventRecord(stop_event, 0));
    gpuErrorCheck(hipEventSynchronize(stop_event));   // block until the event is actually recorded
    gpuErrorCheck(hipEventElapsedTime(&time_memcpy, start_event, stop_event));
    printf("memcopy:\t%.2f\n", time_memcpy);

    // time kernel
    threads=dim3(512, 1);
    blocks=dim3(n / threads.x, 1);
    gpuErrorCheck(hipEventRecord(start_event, 0));
    hipLaunchKernelGGL(init_array, blocks, threads, 0, streams[0], d_a, d_c, niterations);
    gpuErrorCheck(hipEventRecord(stop_event, 0));
    gpuErrorCheck(hipEventSynchronize(stop_event));
    gpuErrorCheck(hipEventElapsedTime(&time_kernel, start_event, stop_event));
    printf("kernel:\t\t%.2f\n", time_kernel);

    //////////////////////////////////////////////////////////////////////
    // time non-streamed execution for reference
    threads=dim3(512, 1);
    blocks=dim3(n / threads.x, 1);
    gpuErrorCheck(hipEventRecord(start_event, 0));

    for (int k = 0; k < nreps; k++)
    {
        hipLaunchKernelGGL(init_array, blocks, threads, 0, 0, d_a, d_c, niterations);
        gpuErrorCheck(hipMemcpy(hAligned_a, d_a, nbytes, hipMemcpyDeviceToHost));
    }

    gpuErrorCheck(hipEventRecord(stop_event, 0));
    gpuErrorCheck(hipEventSynchronize(stop_event));
    gpuErrorCheck(hipEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("non-streamed:\t%.2f\n", elapsed_time / nreps);

    //////////////////////////////////////////////////////////////////////
    // time execution with nstreams streams
    threads=dim3(512,1);
    blocks=dim3(n/(nstreams*threads.x),1);
    memset(hAligned_a, 255, nbytes);     // set host memory bits to all 1s, for testing correctness
    gpuErrorCheck(hipMemset(d_a, 0, nbytes)); // set device memory to all 0s, for testing correctness
    gpuErrorCheck(hipEventRecord(start_event, 0));

    for (int k = 0; k < nreps; k++)
    {
        // asynchronously launch nstreams kernels, each operating on its own portion of data
        for (int i = 0; i < nstreams; i++)
        {
            hipLaunchKernelGGL(init_array, blocks, threads, 0, streams[i], d_a + i *n / nstreams, d_c, niterations);
        }

        // asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
        //   commence executing when all previous CUDA calls in stream x have completed
        for (int i = 0; i < nstreams; i++)
        {
            gpuErrorCheck(hipMemcpyAsync(hAligned_a + i * n / nstreams, d_a + i * n / nstreams, nbytes / nstreams, hipMemcpyDeviceToHost, streams[i]));
        }
    }

    gpuErrorCheck(hipEventRecord(stop_event, 0));
    gpuErrorCheck(hipEventSynchronize(stop_event));
    gpuErrorCheck(hipEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("%d streams:\t%.2f\n", nstreams, elapsed_time / nreps);

    // check whether the output is correct
    printf("-------------------------------\n");
    bool bResults = correct_data(hAligned_a, n, c*nreps*niterations);

    // release resources
    for (int i = 0; i < nstreams; i++)
    {
        gpuErrorCheck(hipStreamDestroy(streams[i]));
    }

    gpuErrorCheck(hipEventDestroy(start_event));
    gpuErrorCheck(hipEventDestroy(stop_event));

    // Free hipHostMalloc or Generic Host allocated memory (from CUDA 4.0)
    FreeHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

    gpuErrorCheck(hipFree(d_a));
    gpuErrorCheck(hipFree(d_c));

    return bResults ? EXIT_SUCCESS : EXIT_FAILURE;
}
