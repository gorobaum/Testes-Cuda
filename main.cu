#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define MS 10

__global__ void multi_kernel(int *ma, int *mb, int *mc) {
  int row = threadIdx.x;
  int column = threadIdx.y;
  row *= MS;
  mc[row+column] = 0;
  int i = 0;
  for (i = 0; i < MS; i++) {
    mc[row+column] += ma[row+i]*mb[column+i*MS];
  }
}

int main() {
  int matrixa[MS][MS], matrixb[MS][MS], matrixc[MS][MS], i, j;
  int *cudamtxa, *cudamtxb, *cudamtxc;
  size_t  size = MS*sizeof(int);
  dim3 threadPerBlock(MS,MS);

  for( i = 0; i < MS; i++ ) {
    for( j = 0; j < MS; j++ ) {
      matrixa[i][j] = i;
      matrixb[i][j] = 2;
    }
  }
  
  printf("mc[0][0] = %d \n", matrixa[2][2]);

  cudaMalloc(&cudamtxa, MS*size);
  cudaMalloc(&cudamtxb, MS*size);
  cudaMalloc(&cudamtxc, MS*size);

  cudaMemcpy(cudamtxa, matrixa, MS*size, cudaMemcpyHostToDevice);
  cudaMemcpy(cudamtxb, matrixb, MS*size, cudaMemcpyHostToDevice);

  multi_kernel<<<1, threadPerBlock>>>(cudamtxa, cudamtxb, cudamtxc);

  cudaMemcpy(matrixc, cudamtxc, MS*size, cudaMemcpyDeviceToHost);

  for( i = 0; i < MS; i++ ) {
    for( j = 0; j < MS; j++ ) {
      printf("%d\t", matrixc[i][j]);
    }
    printf("\n");
  }

  return 0;
}
