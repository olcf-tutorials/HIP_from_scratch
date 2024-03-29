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

  // Warmup run
  gpuErrorCheck(
      hipMemcpy(d_A, A, N * N * sizeof(double), hipMemcpyHostToDevice));
  gpuErrorCheck(
      hipMemcpy(d_B, B, N * N * sizeof(double), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(matrix_multiply, blocks_in_grid, threads_per_block, 0, 0,
                     d_A, d_B, d_C);
  gpuErrorCheck(
      hipMemcpy(C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost));

  // creating two streams and event that we'll synchronize with

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

  // verify results
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

  /* Clean up and output
   * --------------------------------------------------------------*/

  gpuErrorCheck(hipStreamDestroy(stream1));
  gpuErrorCheck(hipStreamDestroy(stream2));
  gpuErrorCheck(hipEventDestroy(datatransfer));

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
