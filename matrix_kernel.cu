#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "matrix_kernel.h"

__global__ void multi_kernel(float *ma, float *mb, float *mc) {
  int row = threadIdx.x;
  int column = threadIdx.y;
  int i = 0;
  
  row *= MS;
  mc[row+column] = 0;
  
  for (i = 0; i < MS; i++) {
    mc[row+column] += ma[row+i]*mb[column+i*MS];
  }
}

void matrixMulti_caller(float **Ma, float **Mb, float *Mc) {
  float *cudamtxa, *cudamtxb, *cudamtxc;
  dim3 threadPerBlock(MS,MS);

  cudaMalloc(&cudamtxa, MS*MS*sizeof(float));
  cudaMalloc(&cudamtxb, MS*MS*sizeof(float));
  cudaMalloc(&cudamtxc, MS*MS*sizeof(float));

  cudaMemcpy(cudamtxa, Ma, MS*MS*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudamtxb, Mb, MS*MS*sizeof(float), cudaMemcpyHostToDevice);

  multi_kernel<<<1, threadPerBlock>>>(cudamtxa, cudamtxb, cudamtxc);

  printf("Matrixc[0][0] = %f\n", Ma[1][1]);

  cudaMemcpy(Mc, cudamtxc, MS*MS*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printf("Matrixc[0][0] = %f\n", Mc[11]);
}
