#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

#define MS 10

int main() {
  int i, j;
  float **matrixa, **matrixb, **matrixc;
  size_t size1 = MS*sizeof(float*),
         size2 = MS*sizeof(float);

  matrixa = malloc(size1);
  matrixb = malloc(size1);
  matrixc = malloc(size1);
  for (i = 0; i < MS; i++) {
    matrixa[i] = malloc(size2);
    matrixb[i] = malloc(size2);
    matrixc[i] = malloc(size2);
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
