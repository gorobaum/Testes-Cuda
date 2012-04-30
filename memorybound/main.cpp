#include <stdio.h>
#include <stdlib.h>
#include "memory.h"

#define MS 10

int main() {
  int i, j;
  float **matrixa, **matrixb, *matrixc;
  
  matrixa = (float**)malloc(MS*sizeof(float*));
  matrixb = (float**)malloc(MS*sizeof(float*));
  matrixc = (float*)malloc(MS*MS*sizeof(float*));
  for (i = 0; i < MS; i++) {
    matrixa[i] = (float*)malloc(MS*sizeof(float));
    matrixb[i] = (float*)malloc(MS*sizeof(float));
    
    for (j = 0; j < MS; j++) {
      matrixa[i][j] = (i+j)*1.0;
      matrixb[i][j] = 2.0;
    }
  }
  
  multiMatrix(matrixa, matrixb, matrixc);  

  for( i = 0; i < MS*MS; i++ ) {
    printf("%f\t", matrixc[i]);
  }
  return 0;
}
