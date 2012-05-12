#include <cstdio>
#include <cstdlib>
#include "memory.h"

#define MS 200

int main() {
  int i;
  double *matrixa, *matrixb;
  
  matrixa = (double*)malloc(MS*MS*sizeof(double));
  matrixb = (double*)malloc(MS*MS*sizeof(double));
  for (i = 0; i < MS*MS; i++) {
      matrixa[i] = (i)*1.0;
      matrixb[i] = 0.0;
  }
  
  //printf("MA[%d] = %f\nMB[%d] = %f\n",0, matrixa[199], 0, matrixb[199]);
  copyMatrix(matrixa, matrixb);  

  for (i = 0; i < MS*MS; i++) {
    printf("Matrixb[%d] = %f\n", i, matrixb[i]);
    getchar();
  }
  printf("\n");
  return 0;
}
