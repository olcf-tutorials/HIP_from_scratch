#include<stdio.h>
#include<stdlib.h>
#include "hip/hip_runtime.h"
#include<time.h>


// Macro for checking errors in CUDA API calls
#define gpuErrorCheck(call)                                                    \
  do {                                                                         \
    hipError_t gpuErr = call;                                                  \
    if (hipSuccess != gpuErr) {                                                \
      printf("GPU Error - %s:%d: '%s' ", __FILE__, __LINE__,   \
             hipGetErrorString(gpuErr));                \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  int* a;
  int megabyte = 1024*1024*1;
  unsigned int size = 512*megabyte*sizeof(int);
  a = (int*)malloc(size);
  for(unsigned int i=0; i<(512*megabyte); i++)
  {
    a[i] = (int)rand()/(int)(RAND_MAX/10);
  }
  
  printf("our character: %d\n", a[23]);

  int* b;
  gpuErrorCheck(hipHostMalloc((void**)&b, size));
  for(unsigned int i=0; i<(512*megabyte); i++)
  {
    b[i] = (int)rand()/(int)(RAND_MAX/10);
  }
  
  printf("our character: %d\n", b[23]);
}
  
  
