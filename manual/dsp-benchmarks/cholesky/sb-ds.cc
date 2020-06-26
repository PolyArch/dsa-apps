#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "temporal.dfg.h"
#include "ss_insts.h"

using std::complex;

#define N _N_

void cholesky(complex<float> *a, complex<float> *L) {
  SS_CONFIG(temporal_config, temporal_size);
  {
    SS_CONST(P_temporal_VAL, *((uint64_t *) a), 1);
    SS_REPEAT_PORT(N - 1);
    SS_RECURRENCE(P_temporal_invsqrt, P_temporal_DIV, 1);
    SS_REPEAT_PORT(N * N / 4);
    SS_RECURRENCE(P_temporal_invpure, P_temporal_V, 1);
    SS_DMA_READ(a + 1, 0, 8 * (N - 1), 1, P_temporal_VEC);
    SS_DMA_WRITE(P_temporal_fin, 8 * N, 8, N - 1, L + N);
    SS_DMA_WRITE(P_temporal_sqrt, 0, 8, 1, L);
    SS_FILL_MODE(STRIDE_ZERO_FILL);
    SS_DMA_READ_STRETCH(a + N + 1, 8 * (N + 1), 8 * (N - 1), -8, N - 1, P_temporal_Z);
    SS_CONFIG_PORT_EXPLICIT((N - 1) * 4, -4);
    SS_DMA_READ(a + 1, 8, 8, N - 1, P_temporal_A);
    SS_DMA_READ_STRETCH(a + 1, 8, 8 * (N - 1), -8, N - 1, P_temporal_B);
    SS_FILL_MODE(NO_FILL);
  }

  int addr = 0;
  int array = 1024;
  for (int i = 1; i < N; ++i) {
    int n = N - i;
    int padded = (n - 1 + (n & 1));

    SS_RECURRENCE(P_temporal_O, P_temporal_VAL, 1);
    SS_SCR_WRITE(P_temporal_O, padded * 8, addr);

    SS_SCR_WRITE(P_temporal_O, n * n / 2 * 8, array);

    SS_WAIT_SCR_WR();

    SS_REPEAT_PORT(n - 1);
    SS_RECURRENCE(P_temporal_invsqrt, P_temporal_DIV, 1);
    SS_SCRATCH_READ(addr, (n - 1) * 8, P_temporal_VEC);
    SS_DMA_WRITE(P_temporal_fin, 8 * N, 8, (n - 1), L + (i + 1) * N + i);
    SS_DMA_WRITE(P_temporal_sqrt, 0, 8, 1, L + i * N + i);

    SS_REPEAT_PORT(n * n / 4);
    SS_RECURRENCE(P_temporal_invpure, P_temporal_V, 1);
    SS_FILL_MODE(STRIDE_ZERO_FILL);
    SS_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * (n - 1), -8, (n - 1), P_temporal_B);
    SS_FILL_MODE(NO_FILL);
    SS_SCR_PORT_STREAM(array, 0, n * n / 2 * 8, 1, P_temporal_Z);
    SS_CONFIG_PORT_EXPLICIT((n - 1) * 4, -4);
    SS_SCR_PORT_STREAM(addr, 8, 8, n - 1, P_temporal_A);

    //SS_GARBAGE(P_temporal_O, n * n / 2);
    //break;

    addr ^= 512;

    int tmp = n * (n - 1) / 2;
    tmp = tmp + tmp % 2;
    SS_CONST(P_temporal_IN, 0, tmp);
    SS_GARBAGE(P_temporal_OUT, tmp);
  }

  SS_WAIT_ALL();
}

#undef N

