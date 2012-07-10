#include <cstdio>
#include <cstdlib>
#include "memory.h"

#define MS 100

int main() {
  int i;
  double *matrixa, *matrixb, *matrixc;
  
  matrixa = (double*)malloc(MS*MS*sizeof(double*));
  matrixb = (double*)malloc(MS*MS*sizeof(double*));
  matrixc = (double*)malloc(MS*MS*sizeof(double*));
  for (i = 0; i < MS*MS; i++) {
      matrixa[i] = (i)*1.0;
      matrixb[i] = 2.0;
  }
  
  multiMatrix(matrixa, matrixb, matrixc);  

  for (i = 0; i < MS*MS; i++) {
    printf("%f\n", matrixc[i]);
  }
  printf("\n");
  return 0;
}
