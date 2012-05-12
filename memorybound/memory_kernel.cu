#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "memory_kernel.h"

__global__ void copy_kernel(double *ma, double *mb) {
  int row = threadIdx.x;
  int column = threadIdx.y;

  row *= MS;
  mb[row+column] = 0.0;
 
  mb[row+column] = ma[row+column];
  if (row == 0 && column == 0) printf("MA[0] = %f\n", mb[row+column]);
}

void copyMatrix_caller(double *Ma, double *Mb) {
  double *cudamtxa, *cudamtxb;
  dim3 threadPerBlock(MS,MS);

  cudaMalloc(&cudamtxa, MS*MS*sizeof(double));
  cudaMalloc(&cudamtxb, MS*MS*sizeof(double));

  cudaMemcpy(cudamtxa, Ma, MS*MS*sizeof(double), cudaMemcpyHostToDevice);

  copy_kernel<<<1, threadPerBlock>>>(cudamtxa, cudamtxb);
  cudaThreadSynchronize();
  cudaMemcpy(Mb, cudamtxb, MS*MS*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}
