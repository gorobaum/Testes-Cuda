#include "matrix.h"
#include "matrix_kernel.h"

void multiMatrix(float **Ma, float **Mb, float *Mc) {
  matrixMulti_caller(Ma, Mb, Mc);
}
