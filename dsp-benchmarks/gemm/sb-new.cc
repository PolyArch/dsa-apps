#include <assert.h>
#include <complex>
#include <iostream>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "compute.dfg.h"
#include "sb_insts.h"

using std::complex;

complex<float> one(1, 0);

void gemm(int n, int m, int p, complex<float> *a, complex<float> *b, complex<float> *c) {
  SB_CONFIG(compute_config, compute_size);

  for (int i = 0; i < n; ++i) {
    SB_CONST(P_compute_C, 0, p);
    SB_RECURRENCE(P_compute_O, P_compute_C, p * (m - 1));
    SB_SCR_PORT_STREAM(0, 0, 8 * m * p, 1, P_compute_B);
    SB_DMA_WRITE(P_compute_O, 0, 8 * p, 1, c + i * p);
    SB_REPEAT_PORT(p / 4);
    SB_DMA_READ(a + i * m, 8, 8, m, P_compute_A);
    //for (int k = 0; k < m; ++k) {
    //  SB_CONST(P_compute_A, *((uint64_t*) (a + i * m + k)), p / 4);
    //}
  }
  SB_WAIT_ALL();

}
