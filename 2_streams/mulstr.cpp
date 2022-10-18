
#include <hip/hip_runtime.h>
#include <hipblas.h>
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
const int num_streams = 8;

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

// int main(int argc, char *argv[]) {
int main() {

  // Set device to GPU 0
  gpuErrorCheck(hipSetDevice(0));
  double *A;
  double *B;
  double *C;

  /* Allocate memory for A, B, C on CPU
   * ----------------------------------------------*/
  // A = (double *)malloc(num_matrices*N * N * sizeof(double));
  // B = (double *)malloc(num_matrices*N * N * sizeof(double));
  // C = (double *)malloc(num_matrices*N * N * sizeof(double));
  gpuErrorCheck(
      hipHostMalloc((void **)&A, (num_matrices * N * N * sizeof(double))));
  gpuErrorCheck(
      hipHostMalloc((void **)&B, (num_matrices * N * N * sizeof(double))));
  gpuErrorCheck(
      hipHostMalloc((void **)&C, (num_matrices * N * N * sizeof(double))));

  /* Set Values for A, B, C on CPU
   * ---------------------------------------------------*/

  // Max size of random double
  double max_value = 10.0;

  // Set A, B, C
  for (int i = 0; i < (num_matrices * N * N); i++) {
    A[i] = (double)rand() / (double)(RAND_MAX / max_value);
    B[i] = (double)rand() / (double)(RAND_MAX / max_value);
    C[i] = 0.0;
  }

  // This was just another convoluted way of filling the arrays
  // that went one matrix at a time instead just doing it in a single loop
  // like above which would've accomplished the same thing
  //  for (int m = 0; m < num_matrices; m++) {
  //    for (int i = 0; i < N; i++) {
  //      for (int j = 0; j < N; j++) {
  //        A[(m * N * N) + (i * N + j)] = (double)rand() / (double)(RAND_MAX /
  //        max_value); B[(m * N * N) + (i * N + j)] = (double)rand() /
  //        (double)(RAND_MAX / max_value); C[(m * N * N) + (i * N + j)] = 0.0;
  //      }
  //    }
  //  }

  /* Allocate memory for d_A, d_B, d_C on GPU
   * ----------------------------------------*/
  double *d_A;
  double *d_B;
  double *d_C;
  gpuErrorCheck(hipMalloc(&d_A, num_matrices * N * N * sizeof(double)));
  gpuErrorCheck(hipMalloc(&d_B, num_matrices * N * N * sizeof(double)));
  gpuErrorCheck(hipMalloc(&d_C, num_matrices * N * N * sizeof(double)));

  /* Perform Matrix Multiply on GPU
   * --------------------------------------------------*/
  dim3 threads_per_block(16, 16, 1);
  dim3 blocks_in_grid(ceil(float(N) / threads_per_block.x),
                      ceil(float(N) / threads_per_block.y), 1);

  // splitting the work across multiple streams

  hipStream_t streams[num_streams];

  for (int i = 0; i < num_streams; i++) {
    gpuErrorCheck(hipStreamCreate(&streams[i]));
  }

  // Warmup run
  gpuErrorCheck(
      hipMemcpy(d_A, A, N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(
      hipMemcpy(d_B, B, N * N * sizeof(double), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0, 0,
                     d_A, d_B, d_C);
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

  // The copy matrices, run kernel, copy result loop
  for (int m = 0; m < num_matrices; m++) {
    int stream_number = m % num_streams;
    gpuErrorCheck(hipMemcpyAsync(&d_A[m * N * N], &A[m * N * N],
                                 N * N * sizeof(double), hipMemcpyHostToDevice,
                                 streams[stream_number]));
    gpuErrorCheck(hipMemcpyAsync(&d_B[m * N * N], &B[m * N * N],
                                 N * N * sizeof(double), hipMemcpyHostToDevice,
                                 streams[stream_number]));

    hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0,
                       streams[stream_number], &d_A[m * N * N], &d_B[m * N * N],
                       &d_C[m * N * N]);
    gpuErrorCheck(hipMemcpyAsync(&C[m * N * N], &d_C[m * N * N],
                                 N * N * sizeof(double), hipMemcpyDeviceToHost,
                                 streams[stream_number]));
  }

  gpuErrorCheck(hipDeviceSynchronize());
  gpuErrorCheck(hipEventRecord(stop));
  gpuErrorCheck(hipEventSynchronize(stop));
  stop_cpu = dtime_usec(0);

  //verify results
   int sample_matrices[10] = {0,    12,      1023,   4000,  54,
                              5555, 1000000, 300234, 90123, 781235};
   double *result_C;
   result_C = (double *)malloc(10 * N * N * sizeof(double));
  
   for (int mat = 0; mat < 10; mat++) {
     int m = sample_matrices[mat];
     double tolerance = 1.0e-12;
     for (int i = 0; i < N; i++) {
       for (int j = 0; j < N; j++) {
         double element = 0.0;
         for (int k = 0; k < N; k++) {
           element +=
               A[(m * N * N) + (i * N + k)] * B[(m * N * N) + (k * N + j)];
         }
  
         if (fabs(C[(m * N * N) + (i * N + j)] - element) > tolerance) {
           printf("For matrix C m%d value of [%d][%d] = %0.14f instead of "
                  "element = %0.14f\n",
                  m, i, j, C[(m * N * N) + (i * N + j)], element);
           exit(1);
         }
       }
     }
   }

  gpuErrorCheck(hipEventElapsedTime(&time_elapsed_hipEvent, start, stop));
  time_elapsed_cpu = (long double)(stop_cpu - start_cpu);
  printf("multiple streams %f milliseconds hipEvent\n", time_elapsed_hipEvent);
  printf("multiple streams %Lf milliseconds cpu\n", time_elapsed_cpu / 1000);

  /* Clean up and output
   * --------------------------------------------------------------*/

  for (int i = 0; i < num_streams; i++) {
    gpuErrorCheck(hipStreamDestroy(streams[i]));
  }

  // Free GPU memory
  gpuErrorCheck(hipFree(d_A));
  gpuErrorCheck(hipFree(d_B));
  gpuErrorCheck(hipFree(d_C));

  // Free CPU memory
  // free(C_fromGPU);
  gpuErrorCheck(hipHostFree(A));
  gpuErrorCheck(hipHostFree(B));
  gpuErrorCheck(hipHostFree(C));

  printf("__SUCCESS__\n");

  return 0;
}
