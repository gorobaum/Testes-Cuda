#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void MatrixCopy (float* MatrixA, float* MatrixB, int rowSize, int columnSize) {
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  int column = blockIdx.y*blockDim.y+threadIdx.y;
  MatrixB[row+column*columnSize] = MatrixA[row+column*columnSize];
}

int main () {
  int row = 4096,
      column = 4096,
      i = 0,
      j = 0;
  dim3 threadPerBlock(16, 16),
       blocksPerGrid(row/threadPerBlock.x, column/threadPerBlock.y);
  size_t size = row*column*sizeof(float);
  float *MatrixA, *MatrixB, *cudaMA, *cudaMB;
  float time;
  cudaEvent_t start, stop;
  
  /* Matrix allocation. */
  MatrixA = (float*)malloc(size);
  MatrixB = (float*)malloc(size);
  
  for (i = 0; i < row; i++)
    for (j = 0; j < column; j++) {
      MatrixA[i*column+j] = i+j;
    }
 
  /* Cuda time counter init. */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Cuda memory allocation. */
  if (cudaMalloc(&cudaMA, size) != cudaSuccess)
      printf("Erro na alocação de recursos!\n");
  if (cudaMalloc(&cudaMB, size) != cudaSuccess)
      printf("Erro na alocação de recursos!\n");

  /* Cuda memory copy. */
  if (cudaMemcpy(cudaMA, MatrixA, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");
  if (cudaMemcpy(cudaMB, MatrixB, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");

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
  printf("%f\n", time);


  if (cudaMemcpy(MatrixB, cudaMB, size, cudaMemcpyDeviceToHost) != cudaSuccess)
      printf("Erro na cópia do Device para o Host!\n");

  
  /*for (i = 0; i < row; i++) {
    for (j = 0; j < column; j++) {
      printf("%.lf\n", MatrixB[i*column+j]);
    }
  }*/
  
  cudaFree(&cudaMA);
  cudaFree(&cudaMB);
  free(MatrixA);
  free(MatrixB);

  return 0;
}
