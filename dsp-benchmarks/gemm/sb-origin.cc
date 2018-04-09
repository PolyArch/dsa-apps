#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "fixed_point.h"
#include "compute.dfg.h"
#include "sb_insts.h"

using std::complex;

void gemm(int n, int m, int p, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  SB_CONFIG(compute_config, compute_size);
#ifdef LATENCY
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
#else
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
#endif
  for (int i = 0; i < n; ++i) {
    SB_CONST(P_compute_C, 0, p / 2);
    SB_RECURRENCE(P_compute_O, P_compute_C, (p / 2) * (m / 2 - 1));
    SB_SCR_PORT_STREAM(0    , 8 * p, 4 * p, m / 2, P_compute_BE);
    SB_SCR_PORT_STREAM(p * 4, 8 * p, 4 * p, m / 2, P_compute_BO);
    SB_DMA_WRITE(P_compute_O, 0, 4 * p, 1, c + i * p);
    for (int k = 0; k < m; k += 2) {
      SB_CONST(P_compute_A, *((uint64_t*) (a + i * m + k)), p / 4);
    }
  }
  SB_WAIT_ALL();
}
