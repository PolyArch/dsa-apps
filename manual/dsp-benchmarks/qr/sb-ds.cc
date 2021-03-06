#include "qr.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "matvec.h"
#include "temporal0.dfg.h"
#include "temporal1.dfg.h"

#define N _N_

complex<float> _one(1, 0);
uint64_t _one_in_bit = *((uint64_t*) &_one);

void qr(complex<float> *a, complex<float> *q, complex<float> *tau) {
  SS_CONFIG(temporal0_config, temporal0_size);

  const int r_trans_spad = 0;
  int w_spad = 8192;

  for (int i = 0; i < N - 1; ++i) {
    int n = N - i;
    int n2 = n / 2 + n % 2;
    int n12 = (n - 1) / 2 + (n - 1) % 2;

    SS_FILL_MODE(STRIDE_ZERO_FILL);
    SS_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_temporal0_M);
    SS_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_temporal0_V);
    SS_CONST(P_temporal0_Coef, _one_in_bit, n12);
    SS_CONST(P_temporal0_reset, 2, n12 - 1);
    SS_CONST(P_temporal0_reset, 1, 1);

    SS_RECURRENCE(P_temporal0_O, P_temporal0_NORM, 1);
    SS_SCRATCH_READ(r_trans_spad, 8, P_temporal0_HEAD);

    SS_DMA_WRITE(P_temporal0_ALPHA, 8, 8, 1, a + i * N + i); //alpha


    SS_SET_ITER(1);
    SS_REPEAT_PORT(n - 1);
    SS_RECURRENCE(P_temporal0_U1INV, P_temporal0_M, 1); //normalize
    SS_DMA_WRITE(P_temporal0_TAU0, 8, 8, 1, tau + i);
    SS_CONST(P_temporal0_Coef, _one_in_bit, n - 1);
    SS_SCR_PORT_STREAM(r_trans_spad + 8, 8, 8, n - 1, P_temporal0_V);
    SS_CONST(P_temporal0_reset, 1, n - 1);

    SS_REPEAT_PORT((n - 1) * n2);
    SS_RECURRENCE(P_temporal0_TAU1, P_temporal0_Coef, 1); //tau
    //SS_GARBAGE(P_temporal0_TAU1, 1); // for debugging
    
    //SS_CONST_SCR(w_spad, _one_in_bit, 1);
    SS_SCR_WRITE(P_temporal0_O, 8 * (n - 1), w_spad + 8);

    SS_WAIT_SCR_WR();
    //SS_WAIT_ALL();

    SS_SCR_PORT_STREAM(w_spad, 0, 8 * n, n - 1, P_temporal0_M);
    SS_SCR_PORT_STREAM(r_trans_spad, 8 * n, 8 * n, n - 1, P_temporal0_V);
    SS_2D_CONST(P_temporal0_reset, 2, n2 - 1, 1, 1, n - 1);


    SS_REPEAT_PORT(n2);
    SS_RECURRENCE(P_temporal0_O, P_temporal0_A, n - 1);

    SS_FILL_MODE(STRIDE_DISCARD_FILL);
    SS_2D_CONST(P_temporal0_Signal, 1, 1, 0, n2 - 1, n - 1);
    SS_SCR_PORT_STREAM(w_spad, 0, 8 * n, n - 1, P_temporal0_B);
    SS_SCR_PORT_STREAM(r_trans_spad, 8 * n, 8 * n, n - 1, P_temporal0_C);
    if (n - 1 == 1) {
      SS_DMA_WRITE(P_temporal0_R, 0, 8, 1, a + N * N - 1);
    } else {
      SS_SCR_WRITE(P_temporal0_R, 8 * (n - 1) * (n - 1), r_trans_spad);
    }
    SS_DMA_WRITE(P_temporal0_FIN, 8, 8, n - 1, a + i * N + i + 1);
    SS_GARBAGE(P_temporal0_R, (n - 1) * (n % 2));

    SS_WAIT_SCR_WR();

    w_spad -= (n - 1) * 8;
  }
  SS_WAIT_ALL();

  w_spad += 8;
  SS_CONFIG(temporal1_config, temporal1_size);
  for (int i = N - 2; i >= 0; --i) {
    int n = N - i;
    int n2 = n / 2 + n % 2;
    int n12 = (n - 1) / 2 + (n - 1) % 2;
    SS_SCR_PORT_STREAM(w_spad, 8, 8, n, P_temporal1_D);
    SS_CONST(P_temporal1_E, *((uint64_t*)(tau + i)), n);
    SS_DMA_WRITE(P_temporal1_FIN, 0, 8 * n, 1, q + i * N + i);

    SS_FILL_MODE(STRIDE_ZERO_FILL);
    SS_SCR_PORT_STREAM(w_spad + 8, 0, 8 * (n - 1), n - 1, P_temporal1_M);
    SS_DMA_READ(q + (i + 1) * N + i + 1, 8 * N, 8 * (n - 1), n - 1, P_temporal1_V);
    SS_CONST(P_temporal1_Coef, *((uint64_t*)(tau + i)), n12 * (n - 1));
    SS_2D_CONST(P_temporal1_reset, 2, n12 - 1, 1, 1, n - 1);
    //SS_GARBAGE(P_temporal1_O, n - 1);

    SS_FILL_MODE(STRIDE_DISCARD_FILL);
    SS_REPEAT_PORT(n2);
    SS_RECURRENCE(P_temporal1_O, P_temporal1_A, n - 1);
    SS_SCR_PORT_STREAM(w_spad, 0, 8 * n, n - 1, P_temporal1_B);
    SS_DMA_READ(q + (i + 1) * N + i, 8 * N, 8 * n, n - 1, P_temporal1_C);
    SS_DMA_WRITE(P_temporal1_Q, 8 * N, 8 * n, n - 1, q + (i + 1) * N + i);
    //SS_GARBAGE(P_temporal1_Q, (n - 1) * (n % 2));

    w_spad += i * 8;
    SS_WAIT_ALL();
  }
  SS_WAIT_ALL();
}

#undef N

