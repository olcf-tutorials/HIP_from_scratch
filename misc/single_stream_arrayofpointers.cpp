#include<time.h>
#include<sys/time.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <stdio.h>

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


const int num_matrices = 1024*1024;
#define N  32
//const int num_streams = 8;

__global__ void matrix_multiply(double *a, double *b, double *c)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row    = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < N && column < N)
    {
        // dot product of row, column
        double element = 0.0;
        for(int i=0; i<N; i++){
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
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}



int main(int argc, char *argv[]) {

  // Set device to GPU 0
  gpuErrorCheck(hipSetDevice(0));
  double *A[num_matrices];
  double *B[num_matrices];
  double *C[num_matrices];

  /* Allocate memory for A, B, C on CPU
   * ----------------------------------------------*/
  for (int i = 0; i < num_matrices; i++) {
   // this works fine, but you lose the benefit of pinned memory
  //  A[i] = (double *)malloc(N * N * sizeof(double));
  //  B[i] = (double *)malloc(N * N * sizeof(double));
  //  C[i] = (double *)malloc(N * N * sizeof(double));

    // this will hit an HipOutOfMemoryError in runtime
    gpuErrorCheck(hipHostMalloc((void**)&A[i], N * N * sizeof(double)));
    gpuErrorCheck(hipHostMalloc((void**)&B[i], N * N * sizeof(double)));
    gpuErrorCheck(hipHostMalloc((void**)&C[i], N * N * sizeof(double)));
  }

  /* Set Values for A, B, C on CPU
   * ---------------------------------------------------*/

  // Max size of random double
  double max_value = 10.0;

  // Set A, B, C
  for (int m = 0; m < num_matrices; m++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        A[m][i * N + j] = (double)rand() / (double)(RAND_MAX / max_value);
        B[m][i * N + j] = (double)rand() / (double)(RAND_MAX / max_value);
        C[m][i * N + j] = 0.0;
      }
    }
  }

  /* Allocate memory for d_A, d_B, d_C on GPU
   * ----------------------------------------*/
  double *d_A[num_matrices];
  double *d_B[num_matrices];
  double *d_C[num_matrices];
  for (int m = 0; m < num_matrices; m++) {
    gpuErrorCheck(hipMalloc(&d_A[m], N * N * sizeof(double)));
    gpuErrorCheck(hipMalloc(&d_B[m], N * N * sizeof(double)));
    gpuErrorCheck(hipMalloc(&d_C[m], N * N * sizeof(double)));

//  gpuErrorCheck(
//      hipMemcpy(d_A[m], A[m], N * N * sizeof(double), hipMemcpyHostToDevice));
//  gpuErrorCheck(
//      hipMemcpy(d_B[m], B[m], N * N * sizeof(double), hipMemcpyHostToDevice));
  }


  /* Perform Matrix Multiply on GPU
   * --------------------------------------------------*/

    dim3 threads_per_block( 16, 16, 1 );
    dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(N) / threads_per_block.y ), 1 );
  
  // Warmup run
  gpuErrorCheck(
      hipMemcpy(d_A[0], A[0], N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(
      hipMemcpy(d_B[0], B[0], N * N * sizeof(double), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0, 0, d_A[0], d_B[0], d_C[0]);

  gpuErrorCheck(
      hipMemcpy(C[0], d_C[0], N * N * sizeof(double), hipMemcpyDeviceToHost));

  // Actual run on a single stream, recording elapsed time
  unsigned long long start_cpu, stop_cpu;
  long double time_elapsed_cpu;
  float time_elapsed_hipEvent;
  hipEvent_t start, stop;

  gpuErrorCheck(hipEventCreate(&start));
  gpuErrorCheck(hipEventCreate(&stop));

  start_cpu = dtime_usec(0);
  gpuErrorCheck(hipEventRecord(start));

  for (int m = 0; m < num_matrices; m++) {
  gpuErrorCheck(
      hipMemcpy(d_A[m], A[m], N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(
      hipMemcpy(d_B[m], B[m], N * N * sizeof(double), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0, 0, d_A[m], d_B[m], d_C[m]);

  gpuErrorCheck(
      hipMemcpy(C[m], d_C[m], N * N * sizeof(double), hipMemcpyDeviceToHost));

 // stop_cpu = dtime_usec(0);
 // time_elapsed_cpu = (long double)(stop_cpu - start_cpu);
 // printf("Time elapsed from start: %Lf  milliseconds as calculated by cpu\n",
 //        time_elapsed_cpu/1000 );
  }

  gpuErrorCheck(hipDeviceSynchronize());
  gpuErrorCheck(hipEventRecord(stop));
  gpuErrorCheck(hipEventSynchronize(stop));
  stop_cpu = dtime_usec(0);

 
  gpuErrorCheck(hipEventElapsedTime(&time_elapsed_hipEvent, start, stop));
  time_elapsed_cpu = (long double)(stop_cpu - start_cpu);
  printf("single stream %f milliseconds hipEvent\n",
         time_elapsed_hipEvent );
  printf("single stream %Lf milliseconds cpu\n",
         time_elapsed_cpu/1000 );

  
  


  /* Clean up and output
   * --------------------------------------------------------------*/


  for (int m = 0; m < num_matrices; m++) {
    // Free GPU memory
    gpuErrorCheck(hipFree(d_A[m]));
    gpuErrorCheck(hipFree(d_B[m]));
    gpuErrorCheck(hipFree(d_C[m]));

    // Free CPU memory
    //free(A[m]);
    //free(B[m]);
    //free(C[m]);
    gpuErrorCheck(hipHostFree(A[m]));
    gpuErrorCheck(hipHostFree(B[m]));
    gpuErrorCheck(hipHostFree(C[m]));
  }

  printf("__SUCCESS__\n");

  return 0;
}
