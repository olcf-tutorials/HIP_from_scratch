#include <hip/hip_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

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

const int num_matrices = 1024 * 1024;
#define N 32
// const int num_streams = 8;

__global__ void matrix_multiply(double *a, double *b, double *c) {
  int column = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < N && column < N) {
    // dot product of row, column
    double element = 0.0;
    for (int i = 0; i < N; i++) {
      element += a[row * N + i] * b[i * N + column];
    }

    c[row * N + column] = element;
  }
}
// host-based timing
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start) {
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

int main(int argc, char *argv[]) {

  // Set device to GPU 0
  gpuErrorCheck(hipSetDevice(0));
  double *A_pageable;
  double *A_pinned;

  /* Allocate memory for A, B for pageable and pinned memory on CPU
   * ----------------------------------------------*/
  A_pageable = (double *)malloc(num_matrices * N * N * sizeof(double));
  gpuErrorCheck(
      hipHostMalloc((void **)&A_pinned, (num_matrices * N * N * sizeof(double))));

  // Max size of random double
  double max_value = 10.0;

  // Set A, B, C
  for (int i = 0; i < (num_matrices * N * N); i++) {
    A_pageable[i] = (double)rand() / (double)(RAND_MAX / max_value);
    A_pinned[i] = (double)rand() / (double)(RAND_MAX / max_value);
  }


  /* Allocate memory for d_A, d_B, d_C on GPU
   * ----------------------------------------*/
  double *d_B;
  gpuErrorCheck(hipMalloc(&d_B, num_matrices * N * N * sizeof(double)));


  dim3 threads_per_block(16, 16, 1);
  dim3 blocks_in_grid(ceil(float(N) / threads_per_block.x),
                      ceil(float(N) / threads_per_block.y), 1);

  // Warmup run
  double *test_pageable;
  double *d_test;
  test_pageable = (double *)malloc(num_matrices * N * N * sizeof(double));
  gpuErrorCheck(hipMalloc(&d_test, num_matrices * N * N * sizeof(double)));
  for (int i = 0; i < (num_matrices * N * N); i++) {
    test_pageable[i] = (double)rand() / (double)(RAND_MAX / max_value);
  }
  gpuErrorCheck(
      hipMemcpy(d_test, test_pageable, N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(
      hipMemcpy(test_pageable, d_test, N * N * sizeof(double), hipMemcpyDeviceToHost));

  // Actual run on a single stream, recording elapsed time
  float time_elapsed_hipEvent;
  hipEvent_t start, stop;

  gpuErrorCheck(hipEventCreate(&start));
  gpuErrorCheck(hipEventCreate(&stop));

  printf("Pinned transfers\n");
  gpuErrorCheck(hipEventRecord(start));

  gpuErrorCheck(
      hipMemcpy(d_B, A_pinned, N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(hipEventRecord(stop));
  gpuErrorCheck(hipEventSynchronize(stop));
  gpuErrorCheck(hipEventElapsedTime(&time_elapsed_hipEvent, start, stop));

  printf("Elapsed time in milliseconds for pinned transfer from Host to Device: hipEvent measurement %f\n", time_elapsed_hipEvent); 

  gpuErrorCheck(hipEventRecord(start));
  gpuErrorCheck(
      hipMemcpy(A_pinned, d_B, N * N * sizeof(double), hipMemcpyDeviceToHost));

  gpuErrorCheck(hipEventRecord(stop));
  gpuErrorCheck(hipEventSynchronize(stop));
  gpuErrorCheck(hipEventElapsedTime(&time_elapsed_hipEvent, start, stop));

  printf("Elapsed time in milliseconds for pinned transfer from Device to Host: hipEvent measurement %f\n", time_elapsed_hipEvent);

  printf("Pageable transfers\n");
  gpuErrorCheck(hipEventRecord(start));

  gpuErrorCheck(
      hipMemcpy(d_B, A_pageable, N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(hipEventRecord(stop));
  gpuErrorCheck(hipEventSynchronize(stop));
  gpuErrorCheck(hipEventElapsedTime(&time_elapsed_hipEvent, start, stop));

  printf("Elapsed time in milliseconds for pageable transfer from Host to Device: hipEvent measurement %f\n", time_elapsed_hipEvent ); 

  gpuErrorCheck(hipEventRecord(start));
  gpuErrorCheck(
      hipMemcpy(A_pageable, d_B, N * N * sizeof(double), hipMemcpyDeviceToHost));

  gpuErrorCheck(hipEventRecord(stop));
  gpuErrorCheck(hipEventSynchronize(stop));
  gpuErrorCheck(hipEventElapsedTime(&time_elapsed_hipEvent, start, stop));

  printf("Elapsed time in milliseconds for pageable transfer from Device to Host: hipEvent measurement %f\n", time_elapsed_hipEvent ); 


  // Free GPU memory
  gpuErrorCheck(hipFree(d_B));

  // Free CPU memory
  free(A_pageable);
  gpuErrorCheck(hipHostFree(A_pinned));

  printf("__SUCCESS__\n");

  return 0;
}
