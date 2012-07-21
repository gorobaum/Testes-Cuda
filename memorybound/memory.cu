#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void MatrixCopy (double* MatrixA, double* MatrixB, int row, int column) {
  int i = threadIdx.x;
  int j = threadIdx.y;

  MatrixB[i+j*row] = MatrixA[i+j*row];
}

int main () {
  int row = 10,
      column = 10,
      i = 0,
      j = 0;
  int blocksPerGrid = 1;
  dim3 threadPerBlock(row, column);
  size_t size = row*column*sizeof(double);
  double *MatrixA, *MatrixB, *cudaMA, *cudaMB;

  MatrixA = (double*)malloc(size);
  MatrixB = (double*)malloc(size);
  
  for (i = 0; i < column; i++)
    for (j = 0; j < row; j++) {
      MatrixA[i+j*row] = i+j*row;
      MatrixB[i+j*row] = 0.0;
    }
  
  /* Cuda memory allocation. */
  cudaMalloc(&cudaMA, size);
  cudaMalloc(&cudaMB, size);

  /* Cuda memory copy. */
  cudaMemcpy(cudaMA, MatrixA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaMB, MatrixB, size, cudaMemcpyHostToDevice);

  /* Cuda kernel call. */
  MatrixCopy<<<blocksPerGrid, threadPerBlock>>>(cudaMA, cudaMB, row, column);

  cudaMemcpy(MatrixB, cudaMB, size, cudaMemcpyDeviceToHost);

  for (i = 0; i < column; i++) {
    for (j = 0; j < row; j++) {
      printf("%lf\t", MatrixB[i+j*row]);
    }
    printf("\n");
  }

  return 0;
}
