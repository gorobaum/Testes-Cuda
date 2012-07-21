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
  int row = 20,
      column = 20,
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
  if (cudaMalloc(&cudaMA, size) != cudaSuccess)
      printf("Erro na alçocação de recursos!\n");
  if (cudaMalloc(&cudaMB, size) != cudaSuccess)
      printf("Erro na alçocação de recursos!\n");

  /* Cuda memory copy. */
  if (cudaMemcpy(cudaMA, MatrixA, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");
  if (cudaMemcpy(cudaMB, MatrixB, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");

  /* Cuda kernel call. */
  MatrixCopy<<<blocksPerGrid, threadPerBlock>>>(cudaMA, cudaMB, row, column);
  cudaDeviceSynchronize();

  if (cudaMemcpy(MatrixB, cudaMB, size, cudaMemcpyDeviceToHost) != cudaSuccess)
      printf("Erro na cópia do Device para o Host!\n");

  for (i = 0; i < column; i++) {
    for (j = 0; j < row; j++) {
      printf("%.1lf ", MatrixB[i+j*row]);
    }
    printf("\n");
  }
  
  cudaFree(&cudaMA);
  cudaFree(&cudaMB);

  return 0;
}
