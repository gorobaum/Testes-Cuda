#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "matrix_kernel.h"

__global__ void multi_kernel(float *ma, float *mb, float *mc) {
  int row = threadIdx.x;
  int column = threadIdx.y;
  row *= MS;
  mc[row+column] = 0;
  int i = 0;
  for (i = 0; i < MS; i++) {
    mc[row+column] += ma[row+i]*mb[column+i*MS];
  }
}

void matrixMulti_caller(float **Ma, float **Mb, float ***Mc) {
  float *cudamtxa, *cudamtxb, *cudamtxc;
  size_t  size = MS*MS*sizeof(float);
  dim3 threadPerBlock(MS,MS);


  cudaMalloc(&cudamtxa, size);
  cudaMalloc(&cudamtxb, size);
  cudaMalloc(&cudamtxc, size);

  cudaMemcpy(cudamtxa, Ma, size, cudaMemcpyHostToDevice);
  cudaMemcpy(cudamtxb, Mb, size, cudaMemcpyHostToDevice);

  multi_kernel<<<1, threadPerBlock>>>(cudamtxa, cudamtxb, cudamtxc);

  cudaMemcpy((*Mc), cudamtxc, size, cudaMemcpyDeviceToHost);

}
