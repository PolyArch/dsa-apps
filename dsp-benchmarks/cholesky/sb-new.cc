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

#define N _N_

void cholesky(complex<float> *a, complex<float> *L) {
  SB_CONFIG(multi_config, multi_size);
  {
    SB_CONST(P_multi_VAL, *((uint64_t *) a), 1);
    SB_REPEAT_PORT(N - 1);
    SB_RECURRENCE(P_multi_invsqrt, P_multi_DIV, 1);
    SB_REPEAT_PORT(N * N / 4);
    SB_RECURRENCE(P_multi_invpure, P_multi_V, 1);
    SB_DMA_READ(a + 1, 0, 8 * (N - 1), 1, P_multi_VEC);
    SB_DMA_WRITE(P_multi_fin, 8 * N, 8, N - 1, L + N);
    SB_DMA_WRITE(P_multi_sqrt, 0, 8, 1, L);
    SB_FILL_MODE(STRIDE_ZERO_FILL);
    SB_DMA_READ_STRETCH(a + N + 1, 8 * (N + 1), 8 * (N - 1), -8, N - 1, P_multi_Z);
    SB_CONFIG_PORT_EXPLICIT((N - 1) * 4, -4);
    SB_DMA_READ(a + 1, 8, 8, N - 1, P_multi_A);
    SB_DMA_READ_STRETCH(a + 1, 8, 8 * (N - 1), -8, N - 1, P_multi_B);
    SB_FILL_MODE(NO_FILL);
  }

  int addr = 0;
  int array = 1024;
  for (int i = 1; i < N; ++i) {
    int n = N - i;
    int padded = (n - 1 + (n & 1));

    SB_RECURRENCE(P_multi_O, P_multi_VAL, 1);
    SB_SCR_WRITE(P_multi_O, padded * 8, addr);

    SB_SCR_WRITE(P_multi_O, n * n / 2 * 8, array);

    SB_WAIT_SCR_WR();

    SB_REPEAT_PORT(n - 1);
    SB_RECURRENCE(P_multi_invpure, P_multi_DIV, 1);
    SB_SCRATCH_READ(addr, (n - 1) * 8, P_multi_VEC);
    SB_DMA_WRITE(P_multi_fin, 8 * N, 8, (n - 1), L + (i + 1) * N);
    SB_DMA_WRITE(P_multi_sqrt, 0, 8, 1, L + i * N + i);

    //SB_GARBAGE(P_multi_invsqrt, 1);
    //break;

    SB_REPEAT_PORT(n * n / 4);
    SB_RECURRENCE(P_multi_invsqrt, P_multi_V, 1);
    SB_FILL_MODE(STRIDE_ZERO_FILL);
    SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * (n - 1), -8, (n - 1), P_multi_B);
    SB_FILL_MODE(NO_FILL);
    SB_SCR_PORT_STREAM(addr, 0, n * n / 2 * 8, 1, P_multi_Z);
    SB_CONFIG_PORT_EXPLICIT((n - 1) * 4, -4);
    SB_SCR_PORT_STREAM(addr, 8, 8, n - 1, P_multi_A);

    //SB_GARBAGE(P_multi_O, n * n / 2);
    //break;

    addr ^= 512;
  }

  SB_WAIT_ALL();
}

#undef N

