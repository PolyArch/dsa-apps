#include "qr.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "matvec.h"
#include "hh.dfg.h"
#include "fused1.dfg.h"
#include "nmlz.dfg.h"
#include "finalize.dfg.h"
#include "fused1.dfg.h"
#include "fused2.dfg.h"

complex<float> _one(1, 0);
complex<float> tempq[1024], w[1024];

void qr(complex<float> *a, complex<float> *q, complex<float> *tau) {
  int N = _N_;
  const int w_spad = 8;
  const int r_trans_spad = N * 8;

  SB_CONTEXT(2);
  SB_CONFIG(fused1_config, fused1_size);

  SB_CONTEXT(4);
  SB_CONFIG(fused2_config, fused2_size);

  SB_CONTEXT(1);
  SB_CONFIG(hh_config, hh_size);
  SB_DMA_READ(a, 8 * N, 8, N, P_hh_A);
  SB_CONST(P_hh_reset, 2, N - 1);
  SB_CONST(P_hh_reset, 1, 1);
  SB_RECURRENCE(P_hh_NORM2_, P_hh__NORM2, 1);
  SB_DMA_READ(a, 8, 8, 1, P_hh_HEAD);
  SB_DMA_WRITE(P_hh_ALPHA, 8, 8, 1, a);
  SB_DMA_WRITE(P_hh_TAUwrite, 8, 8, 1, tau);
  SB_REPEAT_PORT(N - 1);
  SB_RECURRENCE(P_hh_U1inv, P_hh_DIV, 1);
  SB_XFER_RIGHT(P_hh_TAUxfer, P_fused1_IN, 1);
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hh_V);
  SB_XFER_RIGHT(P_hh_O, P_fused1_IN, N - 1);

  SB_CONTEXT(2);
  SB_REPEAT_PORT(N * (N - 1));
  SB_RECURRENCE(P_fused1_OUTspad, P_fused1_TAU, 1);
  SB_SCR_WRITE(P_fused1_OUTspad, 8 * (N - 1), w_spad);
  SB_REPEAT_PORT(N * N);
  SB_XFER_RIGHT(P_fused1_OUTxfer, P_fused2_TAU, 1);
  SB_XFER_RIGHT(P_fused1_OUTxfer, P_fused2_IN, N - 1);

  SB_CONTEXT(4);
  SB_SCR_WRITE(P_fused2_OUTspad, 8 * (N - 1), w_spad);
  SB_DMA_WRITE(P_fused2_OUTmem, 8 * N, 8, N - 1, a + N);
  SB_DMA_SCRATCH_LOAD(&_one, 8, 8, 1, 0);

  SB_CONTEXT(2);
  SB_DMA_SCRATCH_LOAD(&_one, 8, 8, 1, 0);
  SB_WAIT_SCR_WR();
  SB_SCR_PORT_STREAM(0, 0, 8 * N, N - 1, P_fused1_W);
  SB_2D_CONST(P_fused1_reset, 2, N - 1, 1, 1, N - 1);
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i, 8 * N, 8, N, P_fused1_M);
  }

  SB_DMA_READ(a + 1, 8, 8, N - 1, P_fused1_HEAD)
  SB_RECURRENCE(P_fused1_V_VAL, P_fused1_VAL, N - 1);
  SB_DMA_WRITE(P_fused1_FIN, 8, 8, N - 1, a + 1);

  SB_REPEAT_PORT(N - 1);
  SB_RECURRENCE(P_fused1_V_A, P_fused1_A, N - 1);
  SB_SCR_PORT_STREAM(w_spad, 0, 8 * (N - 1), N - 1, P_fused1_B);
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i + N, 8 * N, 8, N - 1, P_fused1_C);
  }

  SB_CONTEXT(4);
  SB_WAIT_SCR_WR();
  SB_REPEAT_PORT(N);
  SB_SCR_PORT_STREAM(0, 8, 8, N, P_fused2_A);
  SB_SCR_PORT_STREAM(0, 0, 8 * N, N, P_fused2_B);
  SB_2D_CONST(P_fused2_C, 1065353216, 1, 0, N, N - 1);
  SB_CONST(P_fused2_C, 1065353216, 1);
  SB_2D_CONST(P_fused2_Signal, 1, 1, 0, N - 1, N);
  SB_DMA_WRITE(P_fused2_TO_MEM, 8 * N, 8, N, q);
  SB_SCR_WRITE(P_fused2_TO_SPAD, 8 * N * (N - 1), r_trans_spad);

  for (int i = 1; i < N - 1; ++i) {
    SB_CONTEXT(2);
    int n = N - i;
    SB_RECURRENCE(P_fused1_RES, P_fused1_IN, n);
    SB_SCR_WRITE(P_fused1_RES, n * (n - 1) * 8, r_trans_spad);
    SB_XFER_LEFT(P_fused1_OUTspad, P_hh_A, n);
    SB_XFER_LEFT(P_fused1_OUTxfer, P_hh_HEAD, 1);
    SB_XFER_LEFT(P_fused1_OUTxfer, P_hh_V, n - 1);
    SB_CONTEXT(1);
    SB_CONST(P_hh_reset, 2, n - 1);
    SB_CONST(P_hh_reset, 1, 1);
    SB_RECURRENCE(P_hh_NORM2_, P_hh__NORM2, 1);
    SB_DMA_WRITE(P_hh_ALPHA, 0, 8, 1, a + i * N + i);
    SB_DMA_WRITE(P_hh_TAUwrite, 0, 8, 1, tau + i);
    SB_XFER_RIGHT(P_hh_TAUxfer, P_fused1_IN, 1);
    SB_REPEAT_PORT(n - 1);
    SB_RECURRENCE(P_hh_U1inv, P_hh_DIV, 1);
    SB_XFER_RIGHT(P_hh_O, P_fused1_IN, n - 1);
    SB_CONTEXT(2);
    SB_REPEAT_PORT(n * (n - 1));
    SB_RECURRENCE(P_fused1_OUTspad, P_fused1_TAU, 1);
    SB_REPEAT_PORT(n * N);
    SB_XFER_RIGHT(P_fused1_OUTxfer, P_fused2_TAU, 1);
    SB_SCR_WRITE(P_fused1_OUTspad, (n - 1) * 8, w_spad);
    SB_XFER_RIGHT(P_fused1_OUTxfer, P_fused2_IN, n - 1);
    SB_CONTEXT(4);
    SB_DMA_WRITE(P_fused2_OUTmem, 8 * N, 8, n - 1, a + (i + 1) * N + i);
    SB_SCR_WRITE(P_fused2_OUTspad, (n - 1) * 8, w_spad);
    SB_CONTEXT(2);
    SB_WAIT_SCR_WR();
    SB_SCR_PORT_STREAM(0, 0, 8 * n, n - 1, P_fused1_W);
    SB_2D_CONST(P_fused1_reset, 2, n - 1, 1, 1, n - 1);
    SB_SCRATCH_READ(r_trans_spad, n * (n - 1) * 8, P_fused1_M);
    SB_RECURRENCE(P_fused1_V_VAL, P_fused1_VAL, n - 1);
    SB_SCR_PORT_STREAM(r_trans_spad, 8 * n, 8, n - 1, P_fused1_HEAD);
    SB_DMA_WRITE(P_fused1_FIN, 8, 8, n - 1, a + i * N + i + 1);
    SB_REPEAT_PORT(n - 1);
    SB_RECURRENCE(P_fused1_V_A, P_fused1_A, n - 1);
    SB_SCR_PORT_STREAM(w_spad, 0, 8 * (n - 1), n - 1, P_fused1_B);
    SB_SCR_PORT_STREAM(r_trans_spad + 8, 8 * n, 8 * (n - 1), (n - 1), P_fused1_C);

    SB_CONTEXT(4);
    SB_WAIT_SCR_WR();
    SB_FILL_MODE(STRIDE_ZERO_FILL);
    SB_SCR_PORT_STREAM(r_trans_spad, 8 * n, 8 * n, N, P_fused2_M);
    SB_SCR_PORT_STREAM(0, 0, 8 * n, N, P_fused2_W);
    SB_2D_CONST(P_fused2_reset, 2, (n + 1) / 2 - 1, 1, 1, N);
    SB_FILL_MODE(NO_FILL);

    SB_REPEAT_PORT(n);
    SB_RECURRENCE(P_fused2_ACC, P_fused2_A, N);
    
    SB_SCR_PORT_STREAM(0, 0, 8 * n, N, P_fused2_B);
    SB_SCRATCH_READ(r_trans_spad, 8 * N * n, P_fused2_C);
    SB_2D_CONST(P_fused2_Signal, 1, 1, 0, n - 1, N);
    SB_DMA_WRITE(P_fused2_TO_MEM, 8 * N, 8, N, q + i);
    if (i == N - 2) {
      SB_DMA_WRITE(P_fused2_TO_SPAD, 8 * N, 8, N, q + i + 1);
    } else {
      SB_SCR_WRITE(P_fused2_TO_SPAD, 8 * N * (n - 1), r_trans_spad);
    }
  }

  SB_CONTEXT(2);
  SB_DMA_WRITE(P_fused1_RES, 8, 8, 1, a + N * N - 1);

  SB_CONTEXT(7);
  SB_WAIT_ALL();
}

