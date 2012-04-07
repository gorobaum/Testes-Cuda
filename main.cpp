#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

#define MS 10

int main() {
  int i, j;
  float **matrixa, **matrixb, **matrixc;

  matrixa = (float**)malloc(MS*sizeof(float*));
  matrixb = (float**)malloc(MS*sizeof(float*));
  matrixc = (float**)malloc(MS*sizeof(float*));
  for (i = 0; i < MS; i++) {
    matrixa[i] = (float*)malloc(MS*sizeof(float));
    matrixb[i] = (float*)malloc(MS*sizeof(float));
    matrixc[i] = (float*)malloc(MS*sizeof(float));
    for (j = 0; j < MS; j++) {
      matrixa[i][j] = i+j*1.0;
          matrixb[i][j] = 2.0;
    }
  }
  
  multiMatrix(matrixa, matrixb, &matrixc);  
 
  for( i = 0; i < MS; i++ ) {
    for( j = 0; j < MS; j++ ) {
      printf("%f\t", matrixc[i][j]);
    }
    printf("\n");
  }

  return 0;
}
