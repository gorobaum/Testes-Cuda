#include <cstdio>
#include <cstdlib>
#include "memory.h"

#define MS 500

int main() {
  int i;
  float *matrixa, *matrixb, *matrixc;
  
  matrixa = (float*)malloc(MS*MS*sizeof(float*));
  matrixb = (float*)malloc(MS*MS*sizeof(float*));
  matrixc = (float*)malloc(MS*MS*sizeof(float*));
  for (i = 0; i < MS*MS; i++) {
      matrixa[i] = (i)*1.0;
      matrixb[i] = 2.0;
  }
  
  multiMatrix(matrixa, matrixb, matrixc);  

  /*for (i = 0; i < MS*MS; i++) {
    printf("%f\t", matrixc[i]);
  }
  printf("\n");*/
  return 0;
}
