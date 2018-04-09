#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "multi.dfg.h"
#include "sb_insts.h"

using std::complex;

void cholesky(complex<float> *a, complex<float> *L) {
  int N = _N_;
  SB_CONTEXT(255);
  SB_CONFIG(multi_config, multi_size);
  {
    SB_CONTEXT(1);
    SB_CONST(P_multi_VAL, *((uint64_t *) a), 1);
    SB_REPEAT_PORT(N - 1);
    SB_RECURRENCE(P_multi_invsqrt, P_multi_DIV, 1);
    SB_REPEAT_PORT((N - 1) * N / 2);
    SB_RECURRENCE(P_multi_invpure, P_multi_V, 1);
    SB_DMA_READ(a + 1, 0, 8 * (N - 1), 1, P_multi_VEC);
    SB_DMA_WRITE(P_multi_fin, 8 * N, 8, N - 1, L + N);
    SB_DMA_WRITE(P_multi_sqrt, 0, 8, 1, L);
    SB_DMA_READ_STRETCH(a + N + 1, 8 * (N + 1), 8 * (N - 1), -8, N - 1, P_multi_Z);
    SB_CONFIG_PORT(N - 1, -1);
    SB_DMA_READ(a + 1, 8, 8, N - 1, P_multi_A);
    SB_DMA_READ_STRETCH(a + 1, 8, 8 * (N - 1), -8, N - 1, P_multi_B);
  }
  int next = 1;
  for (int i = 1, acc = 0, addr = 0; i < N; ++i) {
    int total = N - i - 1;

    SB_CONTEXT(1 << acc);

    SB_XFER_RIGHT(P_multi_O, P_multi_VAL, 1);
    SB_XFER_RIGHT(P_multi_O, P_multi_IN, total);
    SB_XFER_RIGHT(P_multi_O, P_multi_Z, total * (N - i) / 2);

    acc = (acc + 1) & 7;
    SB_CONTEXT(1 << acc);
    SB_DMA_WRITE(P_multi_sqrt, 0, 8, 1, L + i * N + i);
    SB_SCR_WRITE(P_multi_OUT, 8 * total, addr);
    SB_REPEAT_PORT(total);
    SB_RECURRENCE(P_multi_invsqrt, P_multi_DIV, 1);
    SB_REPEAT_PORT((1 + total) * total / 2);
    SB_RECURRENCE(P_multi_invpure, P_multi_V, 1);
    SB_WAIT_SCR_WR();
    SB_SCR_PORT_STREAM(addr, 8, 8, total, P_multi_VEC);
    SB_DMA_WRITE(P_multi_fin, 8 * N, 8, total, L + (i + 1) * N + i);
    SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * total, -8, total, P_multi_B);
    SB_CONFIG_PORT(total, -1);
    SB_SCR_PORT_STREAM(addr, 8, 8, total, P_multi_A);

    //SB_GARBAGE(P_multi_O, (1 + total) * total / 2);
    //SB_WAIT_ALL();
    //return;

    if (acc == 0)
      addr ^= 256;
  }

  SB_CONTEXT(255);
  SB_WAIT_ALL();
}

