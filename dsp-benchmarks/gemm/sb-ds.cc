#include <assert.h>
#include <complex>
#include <iostream>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "temporal.dfg.h"
#include "ss_insts.h"

using std::complex;

complex<float> one(1, 0);

void gemm(int n, int m, int p, complex<float> *a, complex<float> *b, complex<float> *c) {
  SS_CONFIG(temporal_config, temporal_size);

  for (int i = 0; i < n; ++i) {
    SS_CONST(P_temporal_C, 0, p);
    SS_RECURRENCE(P_temporal_O, P_temporal_C, p * (m - 1));
    SS_SCR_PORT_STREAM(0, 0, 8 * m * p, 1, P_temporal_B);
    SS_DMA_WRITE(P_temporal_O, 0, 8 * p, 1, c + i * p);
    SS_REPEAT_PORT(p / 4);
    SS_DMA_READ(a + i * m, 8, 8, m, P_temporal_A);
    //for (int k = 0; k < m; ++k) {
    //  SS_CONST(P_temporal_A, *((uint64_t*) (a + i * m + k)), p / 4);
    //}
  }
  SS_WAIT_ALL();

}
