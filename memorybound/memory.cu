#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void MatrixCopy (double* MatrixA, double* MatrixB, int row, int column) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  MatrixB[i*column+j] = MatrixA[i*column+j];
}

int main () {
  int row = 32,
      column = 32,
      i = 0,
      j = 0;
  dim3 threadPerBlock(16, 16),
       blocksPerGrid(row/threadPerBlock.x+1, column/threadPerBlock.y+1);
  size_t size = row*column*sizeof(double);
  double *MatrixA, *MatrixB, *cudaMA, *cudaMB;
  float time;
  cudaEvent_t start, stop;
  
  /* Matrix allocation. */
  MatrixA = (double*)malloc(size);
  MatrixB = (double*)malloc(size);
  
  for (i = 0; i < row; i++)
    for (j = 0; j < column; j++) {
      MatrixA[i*column+j] = i*column+j;
      MatrixB[i*column+j] = 0.0;
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
  
  /* Cuda time counter init. */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Cuda kernel call. */
  cudaEventRecord(start, 0);
  MatrixCopy<<<blocksPerGrid, threadPerBlock>>>(cudaMA, cudaMB, row, column);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop );

  /* Calculating run time. */
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  printf("Total run time on GPU = %fms\n", time);


  if (cudaMemcpy(MatrixB, cudaMB, size, cudaMemcpyDeviceToHost) != cudaSuccess)
      printf("Erro na cópia do Device para o Host!\n");

  /*
  for (i = 0; i < row; i++) {
    for (j = 0; j < column; j++) {
      printf("%5.lf ", MatrixB[i*column+j]);
    }
    printf("\n");
  }*/
  
  cudaFree(&cudaMA);
  cudaFree(&cudaMB);

  return 0;
}
