#include "memory.h"
#include "memory_kernel.h"

void copyMatrix(double *Ma, double *Mb) {
  copyMatrix_caller(Ma, Mb);
}
