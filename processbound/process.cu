#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void MatrixCopy (double* MatrixA, double* MatrixB, double* MatrixC, int N) {
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int k;

  for (k = 0; k < N; k++ )
    MatrixC[i*N+j] += MatrixA[i*N+k]*MatrixB[k+j*N];
}

int main () {
  int N = 32,
      i = 0,
      j = 0;
  dim3 threadPerBlock(16, 16),
       blocksPerGrid(N/threadPerBlock.x, N/threadPerBlock.y);
  size_t size = N*N*sizeof(double);
  double *MatrixA, *MatrixB, *MatrixC, *cudaMA, *cudaMB, *cudaMC;
  
  /* Matrix allocation. */
  MatrixA = (double*)malloc(size);
  MatrixB = (double*)malloc(size);
  MatrixC = (double*)malloc(size);
 
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      MatrixA[i*N+j] = i*N+j;
      MatrixB[i*N+j] = 2.0;
      MatrixC[i*N+j] = 0.0;
    }
 
  /* Cuda memory allocation. */
  if (cudaMalloc(&cudaMA, size) != cudaSuccess)
      printf("Erro na alçocação de recursos!\n");
  if (cudaMalloc(&cudaMB, size) != cudaSuccess)
      printf("Erro na alçocação de recursos!\n");
  if (cudaMalloc(&cudaMC, size) != cudaSuccess)
      printf("Erro na alçocação de recursos!\n");

  /* Cuda memory copy. */
  if (cudaMemcpy(cudaMA, MatrixA, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");
  if (cudaMemcpy(cudaMB, MatrixB, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");
  if (cudaMemcpy(cudaMC, MatrixC, size, cudaMemcpyHostToDevice) != cudaSuccess)
      printf("Erro na cópia de recursos!\n");

  /* Cuda kernel call. */
  MatrixCopy<<<blocksPerGrid, threadPerBlock>>>(cudaMA, cudaMB, cudaMC, N);
  cudaDeviceSynchronize();

  if (cudaMemcpy(MatrixC, cudaMC, size, cudaMemcpyDeviceToHost) != cudaSuccess)
      printf("Erro na cópia do Device para o Host!\n");

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      printf("%5.lf\n", MatrixC[i*N+j]);
    }
    printf("\n");
  }
  
  cudaFree(&cudaMA);
  cudaFree(&cudaMB);

  return 0;
}
