#include "qr.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "matvec.h"
#include "hhr.dfg.h"
#include "fused.dfg.h"

complex<float> _one(1, 0);
complex<float> buffer[1024];

void qr(complex<float> *a, complex<float> *q, complex<float> *tau) {
  int N = _N_;
  const int w_spad = 8;
  const int r_trans_spad = N * 8;
  const int q_spad = N * 8;

  SB_CONTEXT(2);
  SB_CONFIG(fused_config, fused_size);
  SB_DMA_SCRATCH_LOAD(&_one, 8, 8, 1, 0);

  SB_CONTEXT(1);
  SB_CONFIG(hhr_config, hhr_size);
  SB_DMA_SCRATCH_LOAD(&_one, 8, 8, 1, 0);
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hhr_A);
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hhr_B);
  SB_CONST(P_hhr_Coef, 1065353216, N - 1);
  SB_CONST(P_hhr_reset, 2, N - 2);
  SB_CONST(P_hhr_reset, 1, 1);
  SB_REPEAT_PORT(4);
  SB_RECURRENCE(P_hhr_O, P_hhr_NORM, 1);
  SB_REPEAT_PORT(4);
  SB_DMA_READ(a, 8, 8, 1, P_hhr_HEAD);
  SB_CONST(P_hhr_Inst, 0, 1);
  SB_CONST(P_hhr_Inst, 1, 1);
  SB_CONST(P_hhr_Inst, 2, 1);
  SB_CONST(P_hhr_Inst, 2, 1);

  SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, a); //alpha
  SB_REPEAT_PORT(N - 1);
  SB_RECURRENCE(P_hhr_RES, P_hhr_B, 1); //normalize
  SB_RECURRENCE(P_hhr_RES, P_hhr_IN, 1); //tau
  SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, tau);
  SB_CONST(P_hhr_Coef, 1065353216, N - 1);
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hhr_A);
  SB_CONST(P_hhr_reset, 1, N - 1);

  //SB_DMA_WRITE(P_hhr_O, 8, 8, N - 1, buffer); W test pass!
  //SB_GARBAGE(P_hhr_OUTlocal, 1);              tau test pass!
  SB_REPEAT_PORT(N * (N - 1));
  SB_RECURRENCE(P_hhr_OUTlocal, P_hhr_Coef, 1);
  //SB_GARBAGE(P_hhr_OUTremote, 1); //xfer it!
  SB_REPEAT_PORT(N);
  SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_QTAU, 1);

  SB_RECURRENCE(P_hhr_O, P_hhr_IN, N - 1);

  SB_SCR_WRITE(P_hhr_OUTlocal, 8 * (N - 1), w_spad);
  //SB_GARBAGE(P_hhr_OUTremote, N - 1); //xfer it!
  SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_IN, N - 1);

  SB_CONTEXT(2);
  SB_SCR_WRITE(P_fused_OUT, 8 * (N - 1), w_spad);

  SB_CONTEXT(3);
  SB_WAIT_SCR_WR();

  SB_CONTEXT(2);
  
  SB_SCR_PORT_STREAM(0, 8, 8, N, P_fused_QW);
  SB_CONST(P_fused_QM, 1065353216, N);
  SB_CONST(P_fused_Qreset, 1, N);
  SB_REPEAT_PORT(N);
  SB_RECURRENCE(P_fused_QV, P_fused_QA, N);
  //SB_SCR_PORT_STREAM(0, 8, 8, N, P_fused_QA);
  SB_SCR_PORT_STREAM(0, 0, 8 * N, N, P_fused_QB);
  SB_2D_CONST(P_fused_QC, 1065353216, 1, 0, N, N - 1);
  SB_CONST(P_fused_QC, 1065353216, 1);
  SB_2D_CONST(P_fused_QSignal, 1, 1, 0, N - 1, N);
  SB_DMA_WRITE(P_fused_Q_MEM, 8 * N, 8, N, q);
  SB_SCR_WRITE(P_fused_Q_SPAD, 8 * N * (N - 1), q_spad);

  SB_CONTEXT(1);
  SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * N, N - 1, P_hhr_B);
  SB_2D_CONST(P_hhr_reset, 2, N - 1, 1, 1, N - 1);
  //SB_DMA_WRITE(P_hhr_O, 8, 8, N - 1, buffer + N); test pass!
  //SB_GARBAGE(P_hhr_O, N - 1); test pass!
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i, 8 * N, 8, N, P_hhr_A);
  }

  SB_REPEAT_PORT(N);
  SB_RECURRENCE(P_hhr_O, P_hhr_RA, N - 1);
  SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * N, N - 1, P_hhr_RB);
  SB_2D_CONST(P_hhr_RSignal, 1, 1, 0, N - 1, N - 1);
  SB_DMA_WRITE(P_hhr_R_MEM, 8, 8, N - 1, a + 1);
  SB_SCR_WRITE(P_hhr_R_SPAD, 8 * (N - 1) * (N - 1), r_trans_spad);
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i, 8 * N, 8, N, P_hhr_RC);
  }

  SB_WAIT_SCR_WR();
  for (int i = 1; i < N - 1; ++i) {
    int n = N - i;

    SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_hhr_A);
    SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_hhr_B);
    SB_CONST(P_hhr_Coef, 1065353216, n - 1);
    SB_CONST(P_hhr_reset, 2, n - 2);
    SB_CONST(P_hhr_reset, 1, 1);

    SB_REPEAT_PORT(4);
    SB_RECURRENCE(P_hhr_O, P_hhr_NORM, 1);
    SB_REPEAT_PORT(4);
    SB_SCRATCH_READ(r_trans_spad, 8, P_hhr_HEAD);

    SB_CONST(P_hhr_Inst, 0, 1);
    SB_CONST(P_hhr_Inst, 1, 1);
    SB_CONST(P_hhr_Inst, 2, 1);
    SB_CONST(P_hhr_Inst, 2, 1);

    SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, a + i * N + i); //alpha

    SB_REPEAT_PORT(n - 1);
    SB_RECURRENCE(P_hhr_RES, P_hhr_B, 1); //normalize
    SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, tau + i);
    SB_RECURRENCE(P_hhr_RES, P_hhr_IN, 1); //tau
    SB_CONST(P_hhr_Coef, 1065353216, n - 1);
    SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_hhr_A);
    SB_CONST(P_hhr_reset, 1, n - 1);

    SB_REPEAT_PORT(n * (n - 1));
    SB_RECURRENCE(P_hhr_OUTlocal, P_hhr_Coef, 1);
    //SB_GARBAGE(P_hhr_OUTremote, 1); //xfer it!
    SB_REPEAT_PORT(N * n);
    SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_QTAU, 1);

    SB_RECURRENCE(P_hhr_O, P_hhr_IN, n - 1);

    SB_SCR_WRITE(P_hhr_OUTlocal, 8 * (n - 1), w_spad);
    //SB_GARBAGE(P_hhr_OUTremote, n - 1); //xfer it!
    //SB_DMA_WRITE(P_hhr_OUTremote, 8, 8, n - 1, buffer); test pass!
    SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_IN, n - 1);

    SB_CONTEXT(3);
    SB_WAIT_SCR_WR();

    SB_CONTEXT(2);
    SB_SCR_WRITE(P_fused_OUT, 8 * (n - 1), w_spad);
    SB_WAIT_SCR_WR();
    SB_SCR_PORT_STREAM(q_spad, 8 * n, 8 * n, N, P_fused_QM);
    SB_SCR_PORT_STREAM(0, 0, 8 * n, N, P_fused_QW);
    SB_2D_CONST(P_fused_Qreset, 2, n - 1, 1, 1, N);

    SB_REPEAT_PORT(n);
    SB_RECURRENCE(P_fused_QV, P_fused_QA, N);
    SB_SCR_PORT_STREAM(0, 0, 8 * n, N, P_fused_QB);
    SB_SCRATCH_READ(q_spad, 8 * N * n, P_fused_QC);
    SB_2D_CONST(P_fused_QSignal, 1, 1, 0, n - 1, N);
    SB_DMA_WRITE(P_fused_Q_MEM, 8 * N, 8, N, q + i);
    if (i < N - 2) {
      SB_SCR_WRITE(P_fused_Q_SPAD, 8 * N * (n - 1), q_spad);
    } else {
      SB_DMA_WRITE(P_fused_Q_SPAD, 8 * N, 8, N, q + N - 1);
    }

    SB_CONTEXT(1);
    SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * n, n - 1, P_hhr_B);
    SB_2D_CONST(P_hhr_reset, 2, n - 1, 1, 1, n - 1);
    SB_SCRATCH_READ(r_trans_spad + 8 * n, 8 * (n - 1) * n, P_hhr_A);

    SB_REPEAT_PORT(n);
    SB_RECURRENCE(P_hhr_O, P_hhr_RA, n - 1);
    SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * n, n - 1, P_hhr_RB);
    SB_2D_CONST(P_hhr_RSignal, 1, 1, 0, n - 1, n - 1);
    SB_DMA_WRITE(P_hhr_R_MEM, 8, 8, n - 1, a + i * N + i + 1);
    SB_SCRATCH_READ(r_trans_spad + 8 * n, 8 * (n - 1) * n, P_hhr_RC);
    SB_SCR_WRITE(P_hhr_R_SPAD, 8 * (n - 1) * (n - 1), r_trans_spad);

    SB_WAIT_SCR_WR();

    //SB_CONTEXT(3);
    //SB_WAIT_ALL();
    //SB_CONTEXT(1);
  }
  SB_CONTEXT(1);
  SB_SCRATCH_DMA_STORE(r_trans_spad, 0, 8, 1, a + N * N - 1);
  SB_CONTEXT(3);
  SB_WAIT_ALL();
}

