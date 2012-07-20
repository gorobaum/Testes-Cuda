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
      i = j = 0;
  int threadsPerBlock = row*column,
      blocksPerGrid = 1;
  int* cudaRow, cudaColumn;
  size_t size = row*column*sizeof(double);
  double* MatrixA, MatrixB, cudaMA, cudaMB;

  MatrixA = malloc(size);
  MatrixB = malloc(size);
  
  for (i = 0; i < column; i++)
    for (j = 0; j < row; j++) {
      MatrixA[i+j*row] = i+j*row;
      MatrixB[i+j*row] = 0.0;
    }
  
  /* Cuda memory allocation. */
  cudaMalloc(&cudaMA, size);
  cudaMalloc(&cudaMB, size);
  cudaMalloc(&cudaRow, sizeof(int));
  cudaMalloc(&cudaColumn, sizeof(int));

  /* Cuda memory copy. */
  cudaMemcpy(cudaMA, MatrixA, size, cudaMemCpyHostToDevice);
  cudaMemcpy(cudaMB, MatrixB, size, cudaMemCpyHostToDevice);
  cudaMemcpy(cudaRow, row, sizeof(int), cudaMemCpyHostToDevice);
  cudaMemcpy(cudaColumn, column, sizeof(int), cudaMemCpyHostToDevice);

  /* Cuda kernel call. */
  MatrixCopy<<<blocksPerGrid, threadsPerBlock>>>(cudaMA, cudaMB, cudaRow, cudaColumn);
  return 0;
}
