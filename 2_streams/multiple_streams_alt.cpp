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

//#define gpuErrorCheck(call)                                                    \
//  do {                                                                         \
//    hipError_t gpuErr = call;                                                  \
//    if (hipSuccess != gpuErr) {                                                \
//      printf("GPU Error - %s:%d: '%s' totalsize = %f\n", __FILE__, __LINE__,                 \
//             hipGetErrorString(gpuErr), totalsize/(1024*1024));                                       \
//      exit(0);                                                                 \
//    }                                                                          \
//  } while (0)

const int num_matrices = 1024*1024;
const size_t N = 32;
const int num_streams = 8;
  double totalsize = 0;

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
  double *A;
  double *B;
  double *C;

  /* Allocate memory for A, B, C on CPU
   * ----------------------------------------------*/
    A = (double *)malloc(num_matrices*N * N * sizeof(double));
    B = (double *)malloc(num_matrices*N * N * sizeof(double));
    C = (double *)malloc(num_matrices*N * N * sizeof(double));

  /* Set Values for A, B, C on CPU
   * ---------------------------------------------------*/

  // Max size of random double
  double max_value = 10.0;

  // Set A, B, C
  for (int m = 0; m < num_matrices; m++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        A[(m * N * N ) + (i * N + j)] = (double)rand() / (double)(RAND_MAX / max_value);
        B[(m * N * N) + (i * N + j)] = (double)rand() / (double)(RAND_MAX / max_value);
        C[(m * N * N) + (i * N + j)] = 0.0;
      }
    }
  }

  /* Allocate memory for d_A, d_B, d_C on GPU
   * ----------------------------------------*/
  double *d_A;
  double *d_B;
  double *d_C;
  gpuErrorCheck(hipMalloc(&d_A, num_matrices * N * N * sizeof(double)));
  gpuErrorCheck(hipMalloc(&d_B, num_matrices * N * N * sizeof(double)));
  gpuErrorCheck(hipMalloc(&d_C, num_matrices * N * N * sizeof(double)));


  const double alpha = 1.0;
  const double beta = 0.0;

  /* Perform Matrix Multiply on GPU
   * --------------------------------------------------*/



  // splitting the work across multiple streams

  hipblasHandle_t handle[num_streams];
  hipStream_t streams[num_streams];

  for (int i = 0; i<num_streams; i++) {
    gpuErrorCheck(hipStreamCreate(&streams[i]));
    hipblasCreate(&handle[i]);
  }

  
  // Warmup run
  gpuErrorCheck(
      hipMemcpy(d_A, A, N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(
      hipMemcpy(d_B, B, N * N * sizeof(double), hipMemcpyHostToDevice));
 hipblasStatus_t status =
     hipblasDgemm(handle[0], HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, &alpha,
                  d_A, N, d_B, N, &beta, d_C, N);
   if (status != HIPBLAS_STATUS_SUCCESS) {
     printf("hipblasDgemm failed with code %d\n", status);
     return EXIT_FAILURE;
 }
  gpuErrorCheck(
      hipMemcpy(C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost));
  
  // Actual run on multiple streams, recording elapsed time
  unsigned long long start_cpu, stop_cpu;
  long double time_elapsed_cpu;
  float time_elapsed_hipEvent;
  hipEvent_t start, stop;

  gpuErrorCheck(hipEventCreate(&start));
  gpuErrorCheck(hipEventCreate(&stop));

  start_cpu = dtime_usec(0);
  gpuErrorCheck(hipEventRecord(start));

  for (int m = 0; m < num_matrices; m++) {
    int stream_number = m % num_streams;
    gpuErrorCheck(
        hipMemcpyAsync(&d_A[m*N*N], &A[m*N*N], N * N * sizeof(double), hipMemcpyHostToDevice,streams[stream_number] ));
    gpuErrorCheck(
        hipMemcpyAsync(&d_B[m*N*N], &B[m*N*N], N * N * sizeof(double), hipMemcpyHostToDevice, streams[stream_number]));
  
    status = hipblasSetStream(handle[0], streams[stream_number]); 
    status = hipblasDgemm(handle[stream_number], HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, &alpha,
                     &d_A[m*N*N], N, &d_B[m*N*N], N, &beta, &d_C[m*N*N], N);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      printf("hipblasDgemm failed with code %d\n", status);
      return EXIT_FAILURE;
    }
    gpuErrorCheck(
      hipMemcpyAsync(&C[m*N*N], &d_C[m*N*N], N * N * sizeof(double), hipMemcpyDeviceToHost, streams[stream_number]));

//  stop_cpu = dtime_usec(0);
//  time_elapsed_cpu = (long double)(stop_cpu - start_cpu);
//  printf("Time elapsed from start: %Lf  milliseconds as calculated by cpu\n",
//         time_elapsed_cpu/1000 );
  }

  gpuErrorCheck(hipDeviceSynchronize());
  gpuErrorCheck(hipEventRecord(stop));
  gpuErrorCheck(hipEventSynchronize(stop));
  stop_cpu = dtime_usec(0);

 
  gpuErrorCheck(hipEventElapsedTime(&time_elapsed_hipEvent, start, stop));
  time_elapsed_cpu = (long double)(stop_cpu - start_cpu);

  printf("multiple stream %f milliseconds hipEvent\n",
         time_elapsed_hipEvent );
  printf("multiple stream %Lf milliseconds cpu\n",
         time_elapsed_cpu/1000 );

  /* Clean up and output
   * --------------------------------------------------------------*/


  for (int i=0; i<num_streams; i++) {
    gpuErrorCheck(hipStreamDestroy(streams[i]));
    hipblasDestroy(handle[i]);
  }

    // Free GPU memory
    gpuErrorCheck(hipFree(d_A));
    gpuErrorCheck(hipFree(d_B));
    gpuErrorCheck(hipFree(d_C));

    // Free CPU memory
    // free(C_fromGPU);
    free(A);
    free(B);
    free(C);

  printf("__SUCCESS__\n");

  return 0;
}
