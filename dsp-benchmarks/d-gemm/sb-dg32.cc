#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "ss-config/fixed_point.h"
#include "dgra32.dfg.h"
#include "sb_insts.h"

using std::complex;

#define PI 3.14159265358979303

void gemm(int n, int m, int p, complex<float> *a, complex<float> *b, complex<float> *c) {
  SB_CONFIG(dgra32_config, dgra32_size);
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p, 1, 0);
  SB_WAIT_SCR_WR();
  for (int i = 0; i < n; ++i) {
    SB_CONST(P_dgra32_C, 0, p);
    SB_RECURRENCE(P_dgra32_O, P_dgra32_C, p * (m - 1));
    SB_SCR_PORT_STREAM(0    , 8 * p, 8 * p, m, P_dgra32_B);
    SB_DMA_WRITE(P_dgra32_O, 0, 8 * p, 1, c + i * p);
    SB_REPEAT_PORT(p / 4);
    SB_DMA_READ(a + i * m, 0, 8 * m, 1, P_dgra32_A);
    /*for (int k = 0; k < m; k += 2) {
      SB_DMA_READ(a + i * m + k, 0, 8, p / 4, P_compute_A);
    }*/
  }
  SB_WAIT_ALL();
}
