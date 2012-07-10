#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "memory_kernel.h"

__global__ void multi_kernel(double *ma, double *mb, double *mc) {
  int row = threadIdx.x;
  int column = threadIdx.y;
  int i = 0;
  
  row *= MS;
  mc[row+column] = 0;
  
  for (i = 0; i < MS; i++) {
    mc[row+column] += ma[row+i]*mb[column+i*MS];
  }
}

void matrixMulti_caller(double *Ma, double *Mb, double *Mc) {
  double *cudamtxa, *cudamtxb, *cudamtxc;
  dim3 threadPerBlock(MS,MS);
  size_t free, total;
  
  cudaDeviceReset();

  cudaMemGetInfo(&free, &total);
  printf("GPU Memory Info -\n");
  printf("GPU Free Memory = %d MB\n", free/(1024*1024));
  printf("GPU Total Memory = %d MB\n", total/(1024*1024));
  getchar();

  cudaMalloc(&cudamtxa, MS*MS*sizeof(double));
  cudaMalloc(&cudamtxb, MS*MS*sizeof(double));
  cudaMalloc(&cudamtxc, MS*MS*sizeof(double));

  cudaMemcpy(cudamtxa, Ma, MS*MS*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cudamtxb, Mb, MS*MS*sizeof(double), cudaMemcpyHostToDevice);

  multi_kernel<<<1, threadPerBlock>>>(cudamtxa, cudamtxb, cudamtxc);


  cudaMemcpy(Mc, cudamtxc, MS*MS*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}
