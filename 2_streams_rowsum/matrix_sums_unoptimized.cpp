#include "hip/hip_runtime.h"
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

// error checking macro
#define hipErrorCheck(call)                                                    \
  do {                                                                         \
    hipError_t hipErr = call;                                                  \
    if (hipSuccess != hipErr) {                                                \
      printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__,                  \
             hipGetErrorString(hipErr));                                       \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

const size_t DSIZE = 16384; // matrix side dimension
const int block_size = 256; // HIP maximum is 1024

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


// empty kernel to jump start the GPU and get correct rocprof outputs
__global__ void init_kernel(double* a, double* b) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tidx < DSIZE) 
    b[tidx] = a[tidx] * b[tidx] + b[tidx] * a[tidx];

  
}

bool validate(float *data, size_t sz) {
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {
      printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i],
             (float)sz);
      return false;
    }
  return true;
}

int main() {

  srand(time(NULL));
  // running init_kernel to initialize GPU
  double* a = (double*)malloc(sizeof(double)*DSIZE);
  double* b = (double*)malloc(sizeof(double)*DSIZE);
  double* d_a;
  double* d_b;
  double max_value = 10.0;
  for(int i = 0 ; i< DSIZE; i++) {
    a[i] = (double)rand() / (double)(RAND_MAX / max_value);
    b[i] = (double)rand() / (double)(RAND_MAX / max_value);
  }
    
  hipErrorCheck(hipMalloc(&d_a, sizeof(double)*DSIZE));
  hipErrorCheck(hipMalloc(&d_b, sizeof(double)*DSIZE));
  hipErrorCheck(hipMemcpy(d_a, a, sizeof(double)*DSIZE, hipMemcpyHostToDevice));
  hipErrorCheck(hipMemcpy(d_b, b, sizeof(double)*DSIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(init_kernel, dim3(DSIZE/block_size), dim3(block_size), 0, 0, d_a, d_b);
  // Check for errors in kernel launch (e.g. invalid execution configuration
  // paramters)
  hipErrorCheck(hipGetLastError());
  // Check for errors on the GPU after control is returned to CPU
  hipErrorCheck(hipDeviceSynchronize());
  

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE * DSIZE]; // allocate space for data in host memory
  h_sums = new float[DSIZE]();

  for (int i = 0; i < DSIZE * DSIZE; i++) // initialize matrix in host memory
    h_A[i] = 1.0f;

  hipErrorCheck(hipMalloc(
      &d_A, DSIZE * DSIZE * sizeof(float))); // allocate device space for A
  hipErrorCheck(hipMalloc(
      &d_sums,
      DSIZE * sizeof(float))); // allocate device space for vector d_sums

  // copy matrix A to device:
  hipErrorCheck(hipMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float),
                          hipMemcpyHostToDevice));

  int grid_size = (DSIZE + block_size - 1) / block_size;

  hipLaunchKernelGGL(row_sums, dim3(grid_size), dim3(block_size), 0, 0, d_A,
                     d_sums, DSIZE);
  // Check for errors in kernel launch (e.g. invalid execution configuration
  // paramters)
  hipErrorCheck(hipGetLastError());
  // Check for errors on the GPU after control is returned to CPU
  hipErrorCheck(hipDeviceSynchronize());

  // copy vector sums from device to host:
  hipErrorCheck(
      hipMemcpy(h_sums, d_sums, DSIZE * sizeof(float), hipMemcpyDeviceToHost));

  if (!validate(h_sums, DSIZE))
    return -1;
  printf("row sums correct!\n");

  return 0;
}