#include <cstdio>
#include <cstdlib>
#include "cuda.h"
#include "cuda_runtime.h"
#include "memory_kernel.h"

__global__ void copy_kernel(double *ma, double *mb) {
  int row = threadIdx.x;
  int column = threadIdx.y;
  
  row *= MS;
  mb[row+column] = 0.0;
}  

void copyMatrix_caller(double *Ma, double *Mb) {
  double *cudamtxa, *cudamtxb;
  dim3 threadPerBlock(MS,MS);
  int i;

  for (i = 0; i < MS*MS; i++) if (Mb[i] != 0.0) printf("Erro!%d\n", i);
  if (cudaMalloc(&cudamtxa, MS*MS*sizeof(double)) == cudaErrorMemoryAllocation) {
    printf("Erro na alocação de memória do CUDA.\n");
    exit(0);
  }
  if (cudaMalloc(&cudamtxb, MS*MS*sizeof(double)) == cudaErrorMemoryAllocation) {
    printf("Erro na alocação de memória do CUDA.\n");
    exit(0);
  }

  if (cudaMemcpy(cudamtxa, Ma, MS*MS*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("erro na copia de memória.\n");
    exit(0);
  }

  copy_kernel<<<1, threadPerBlock>>>(cudamtxa, cudamtxb);
  if (cudaMemcpy(Ma, cudamtxa, MS*MS*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
    printf("erro na copia de memória.\n");
    exit(0);
  }
  for (i = 0; i < MS*MS; i++) printf("Ma[%d] = %f\n", i, Ma[i]);
  cudaDeviceSynchronize();
}
