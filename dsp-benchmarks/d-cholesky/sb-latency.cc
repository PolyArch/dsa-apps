#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "multi2.dfg.h"
#include "sb_insts.h"

using std::complex;

void cholesky(complex<float> *a, complex<float> *L) { int N = _N_;
  SB_CONTEXT(255);
  SB_CONFIG(multi2_config, multi2_size);
  {
    SB_CONTEXT(1);
    SB_CONST(P_multi2_VAL, *((uint64_t *) a), 1);
    SB_REPEAT_PORT(N - 1);
    SB_RECURRENCE(P_multi2_invsqrt, P_multi2_DIV, 1);
    SB_REPEAT_PORT(N * N / 4);
    SB_RECURRENCE(P_multi2_invpure, P_multi2_V, 1);
    SB_DMA_READ(a + 1, 0, 8 * (N - 1), 1, P_multi2_VEC);
    SB_DMA_WRITE(P_multi2_fin, 8 * N, 8, N - 1, L + N);
    SB_DMA_WRITE(P_multi2_sqrt, 0, 8, 1, L);
    SB_FILL_MODE(STRIDE_ZERO_FILL);
    SB_DMA_READ_STRETCH(a + N + 1, 8 * (N + 1), 8 * (N - 1), -8, N - 1, P_multi2_Z);
    SB_CONFIG_PORT_EXPLICIT((N - 1) * 4, -4);
    SB_DMA_READ(a + 1, 8, 8, N - 1, P_multi2_A);
    SB_DMA_READ_STRETCH(a + 1, 8, 8 * (N - 1), -8, N - 1, P_multi2_B);
    SB_FILL_MODE(NO_FILL);
  }
  int next = 1;
  for (int i = 1, acc = 0, addr = 0; i < N; ++i) {
    //int total = N - i - 1;
    int n = N - i;
    int padded = (n - 1 + (n & 1));

    SB_CONTEXT(1 << acc);

    SB_XFER_RIGHT(P_multi2_O, P_multi2_VAL, 1);
    SB_XFER_RIGHT(P_multi2_O, P_multi2_IN, padded);
    SB_XFER_RIGHT(P_multi2_O, P_multi2_Z, n * n / 2);

    acc = (acc + 1) & 7;
    SB_CONTEXT(1 << acc);

    SB_SCR_WRITE(P_multi2_OUT, 8 * padded, addr);
    SB_WAIT_SCR_WR();

    SB_REPEAT_PORT(n - 1);
    SB_RECURRENCE(P_multi2_invsqrt, P_multi2_DIV, 1);
    SB_SCR_PORT_STREAM(addr, 8, 8, n - 1, P_multi2_VEC);
    SB_DMA_WRITE(P_multi2_sqrt, 0, 8, 1, L + i * N + i);
    SB_DMA_WRITE(P_multi2_fin, 8 * N, 8, n - 1, L + (i + 1) * N + i);

    SB_REPEAT_PORT(n * n / 4);
    SB_RECURRENCE(P_multi2_invpure, P_multi2_V, 1);
    SB_FILL_MODE(STRIDE_ZERO_FILL);
    SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * (n - 1), -8, (n - 1), P_multi2_B);
    SB_FILL_MODE(NO_FILL);
    SB_CONFIG_PORT_EXPLICIT((n - 1) * 4, -4);
    SB_SCR_PORT_STREAM(addr, 8, 8, n - 1, P_multi2_A);

    //SB_GARBAGE(P_multi2_O, (1 + total) * total / 2);
    //SB_WAIT_ALL();
    //return;

    if (acc == 0)
      addr ^= 256;
  }

  SB_CONTEXT(255);
  SB_WAIT_ALL();
}

