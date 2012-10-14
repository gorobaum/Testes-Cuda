#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void MatrixCopy (float* MatrixA, float* MatrixB, float* MatrixC, int N) {
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int k;
  MatrixC[i*N+j] = 0;
  for (k = 0; k < N; k++ )
    MatrixC[i*N+j] += MatrixA[i*N+k]*MatrixB[k*N+j];
}

int main () {
  int N = 500,
      i = 0,
      j = 0;
  dim3 threadPerBlock(32, 32),
       blocksPerGrid(N/threadPerBlock.x+1, N/threadPerBlock.y+1);
  size_t size = N*N*sizeof(float);
  float *MatrixA, *MatrixB, *MatrixC, *cudaMA, *cudaMB, *cudaMC;
  float time;
  cudaEvent_t start, stop;
  
  /* Matrix allocation. */
  MatrixA = (float*)malloc(size);
  MatrixB = (float*)malloc(size);
  MatrixC = (float*)malloc(size);
 
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      MatrixA[i*N+j] = i+j;
      MatrixB[i*N+j] = 2.0;
      MatrixC[i*N+j] = 0.0;
    }
 
  /* Cuda memory allocation. */
  if (cudaMalloc(&cudaMA, size) != cudaSuccess)
      printf("Erro na alocação de recursos!\n");
  if (cudaMalloc(&cudaMB, size) != cudaSuccess)
      printf("Erro na alocação de recursos!\n");
  if (cudaMalloc(&cudaMC, size) != cudaSuccess)
      printf("Erro na alocação de recursos!\n");

  /* Cuda memory copy. */
  if (cudaMemcpy(cudaMA, MatrixA, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");
  if (cudaMemcpy(cudaMB, MatrixB, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");
  if (cudaMemcpy(cudaMC, MatrixC, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");

  /* Cuda time counter init. */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Cuda kernel call. */
  cudaEventRecord(start, 0);
  MatrixCopy<<<blocksPerGrid, threadPerBlock>>>(cudaMA, cudaMB, cudaMC, N);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop );

  /* Calculating run time. */
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  printf("Total run time on GPU = %fms\n", time);

  if (cudaMemcpy(MatrixC, cudaMC, size, cudaMemcpyDeviceToHost) != cudaSuccess)
      printf("Erro na cópia do Device para o Host!\n");

  
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      printf("C[%d][%d] = %f\n", i, j, MatrixC[i*N+j]);
    }
  }
  
  cudaFree(&cudaMA);
  cudaFree(&cudaMB);

  return 0;
}
