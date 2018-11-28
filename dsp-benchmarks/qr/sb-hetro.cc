#include "qr.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "matvec.h"
#include "rdfg.dfg.h"
#include "qdfg.dfg.h"

#define N _N_

complex<float> _one(1, 0);
uint64_t _one_in_bit = *((uint64_t*) &_one);

void qr(complex<float> *a, complex<float> *q, complex<float> *tau) {
  SB_CONFIG(rdfg_config, rdfg_size);

  const int r_trans_spad = 0;
  int w_spad = 8192;

  SB_FILL_MODE(NO_FILL);
  for (int i = 0; i < N - 1; ++i) {
    int n = N - i;
    int n2 = n / 2 + n % 2;
    int n12 = (n - 1) / 2 + (n - 1) % 2;

    {
      int pad = (n - 1) & 1;
      SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_rdfg_M); SB_CONST(P_rdfg_M, 0, pad);
      SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_rdfg_V); SB_CONST(P_rdfg_V, 0, pad);
      SB_CONST(P_rdfg_Coef, _one_in_bit, n12);
      SB_CONST(P_rdfg_reset, 2, n12 - 1);
      SB_CONST(P_rdfg_reset, 1, 1);
    }

    SB_RECURRENCE(P_rdfg_O, P_rdfg_NORM, 1);
    SB_SCRATCH_READ(r_trans_spad, 8, P_rdfg_HEAD);

    SB_DMA_WRITE(P_rdfg_ALPHA, 8, 8, 1, a + i * N + i); //alpha


    uint64_t tau0;
    SB_RECV(P_rdfg_U1INV, tau0);
    SB_CONST(P_rdfg_M, tau0, (n - 1) * 2);

    SB_DMA_WRITE(P_rdfg_TAU0, 8, 8, 1, tau + i);

    SB_CONST(P_rdfg_Coef, _one_in_bit, n - 1);
    SB_CONST(P_rdfg_reset, 1, n - 1);
    for (int j = 0; j < n - 1; ++j) {
      SB_SCR_PORT_STREAM(r_trans_spad + 8 * j, 8, 8, 1, P_rdfg_V);
      SB_CONST(P_rdfg_V, 0, 1);
    }

    SB_REPEAT_PORT((n - 1) * n2);
    SB_RECURRENCE(P_rdfg_TAU1, P_rdfg_Coef, 1); //tau
    //SB_GARBAGE(P_rdfg_TAU1, 1); // for debugging
    
    SB_CONST_SCR(w_spad, _one_in_bit, 1);
    SB_SCR_WRITE(P_rdfg_O, 8 * (n - 1), w_spad + 8);

    SB_WAIT_SCR_WR();
    //SB_WAIT_ALL();

    SB_REPEAT_PORT(n2);
    SB_RECURRENCE(P_rdfg_O, P_rdfg_A, n - 1);
    SB_2D_CONST(P_rdfg_Signal, 1, 1, 0, n2 - 1, n - 1);
    SB_DMA_WRITE(P_rdfg_FIN, 8 * N, 8, n - 1, a + (i + 1) * N + i);
    {
      int pad = n & 1;
      SB_2D_CONST(P_rdfg_reset, 2, n2 - 1, 1, 1, n - 1);
      for (int j = 0; j < n - 1; ++j) {
        SB_SCR_PORT_STREAM(w_spad, 0, 8 * n, 1, P_rdfg_M); SB_CONST(P_rdfg_M, 0, pad);
        SB_SCR_PORT_STREAM(r_trans_spad + j * 8 * n, 0, 8 * n, 1, P_rdfg_V); SB_CONST(P_rdfg_V, 0, pad);
        SB_SCR_PORT_STREAM(w_spad, 0, 8 * n, 1, P_rdfg_B); SB_CONST(P_rdfg_B, 0, pad);
        SB_SCR_PORT_STREAM(r_trans_spad + j * 8 * n, 0, 8 * n, 1, P_rdfg_C); SB_CONST(P_rdfg_C, 0, pad);
        SB_SCR_WRITE(P_rdfg_R, 8 * (n - 1), r_trans_spad); SB_GARBAGE(P_rdfg_R, pad);
      }
    }

    SB_WAIT_SCR_WR();

    w_spad -= (n - 1) * 8;
  }
  SB_WAIT_ALL();

  w_spad += 8;
  SB_CONFIG(qdfg_config, qdfg_size);
  for (int i = N - 2; i >= 0; --i) {
    int n = N - i;
    int n2 = n / 2 + n % 2;
    int n12 = (n - 1) / 2 + (n - 1) % 2;
    SB_SCR_PORT_STREAM(w_spad, 8, 8, n, P_qdfg_D);
    SB_CONST(P_qdfg_E, *((uint64_t*)(tau + i)), n);
    SB_DMA_WRITE(P_qdfg_FIN, 0, 8 * n, 1, q + i * N + i);

    SB_CONST(P_qdfg_Coef, *((uint64_t*)(tau + i)), n12 * (n - 1));
    SB_2D_CONST(P_qdfg_reset, 2, n12 - 1, 1, 1, n - 1);
    SB_REPEAT_PORT(n2);
    SB_RECURRENCE(P_qdfg_O, P_qdfg_A, n - 1);
    {
      int pad1 = (n - 1) & 1;
      int pad0 = n & 1;
      for (int j = 0; j < n - 1; ++j) {
        SB_SCR_PORT_STREAM(w_spad + 8, 0, 8 * (n - 1), 1, P_qdfg_M); SB_CONST(P_qdfg_M, 0, pad1);
        SB_DMA_READ(q + (i + 1 + j) * N + i + 1, 8 * N, 8 * (n - 1), 1, P_qdfg_V); SB_CONST(P_qdfg_V, 0, pad1);
        SB_SCR_PORT_STREAM(w_spad, 0, 8 * n, 1, P_qdfg_B); SB_CONST(P_qdfg_B, 0, pad0);
        SB_DMA_READ(q + (j + i + 1) * N + i, 8 * N, 8 * n, 1, P_qdfg_C); SB_CONST(P_qdfg_C, 0, pad0);
        SB_DMA_WRITE(P_qdfg_Q, 8 * N, 8 * n, 1, q + (i + 1) * N + i); SB_GARBAGE(P_qdfg_Q, pad0);
      }
    }

    w_spad += i * 8;
    SB_WAIT_ALL();
  }
  SB_WAIT_ALL();
}

#undef N

