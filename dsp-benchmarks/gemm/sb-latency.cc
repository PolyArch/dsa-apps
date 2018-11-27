#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "compute.dfg.h"
#include "sb_insts.h"

using std::complex;

complex<float> one(1, 0);

void gemm(int n, int m, int p, complex<float> *a, complex<float> *b, complex<float> *c) {
  SB_CONTEXT(255);
  SB_CONFIG(compute_config, compute_size);

  int resudo = n & 7;
  int _n = n + (resudo != 0) * (8 - resudo);
  for (int io = 0; io < _n; io += 8) {
    SB_CONTEXT_I(255);
    SB_CONST(P_compute_C, 0, p);
    SB_RECURRENCE(P_compute_O, P_compute_C, p * (m - 1));
    SB_SCR_PORT_STREAM(0    , 0, 8 * m * p, 1, P_compute_B);

    SB_STRIDE(8, 8);

    complex<float> *_c = c + io * p;
    complex<float> *_a = a + io * m;

    SB_CONTEXT_OFFSET(255, p);
    SB_DMA_WRITE_SIMP(P_compute_O, p, _c);
    SB_CONTEXT_OFFSET(255, m)
    SB_REPEAT_PORT(p / 4);
    SB_DMA_READ_SIMP(_a, m, P_compute_A);
    SB_CONTEXT_OFFSET(255, 0);
  }

  /*if (resudo) {

    SB_CONTEXT_I(15);
    SB_CONST(P_compute_C, 0, p / 2);
    SB_RECURRENCE(P_compute_O, P_compute_C, (p / 2) * (m / 2 - 1));
    SB_SCR_PORT_STREAM(0    , 8 * p, 4 * p, m / 2, P_compute_BE);
    SB_SCR_PORT_STREAM(p * 4, 8 * p, 4 * p, m / 2, P_compute_BO);

    SB_STRIDE(8, 8);

    complex<float> *_c = c + io * p;
    complex<float> *_a = a + io * m;

    SB_CONTEXT_OFFSET(15, p);
    SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
    SB_CONTEXT_OFFSET(15, m)
    SB_REPEAT_PORT(p / 4);
    SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
    SB_CONTEXT_OFFSET(15, 0);

  }*/

  SB_CONTEXT(255);
  SB_WAIT_ALL();
}
