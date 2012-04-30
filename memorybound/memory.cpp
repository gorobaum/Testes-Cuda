#include "memory.h"
#include "memory_kernel.h"

void multiMatrix(float *Ma, float *Mb, float *Mc) {
  matrixMulti_caller(Ma, Mb, Mc);
}
