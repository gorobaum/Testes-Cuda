#include "memory.h"
#include "memory_kernel.h"

void multiMatrix(double *Ma, double *Mb, double *Mc) {
  matrixMulti_caller(Ma, Mb, Mc);
}
