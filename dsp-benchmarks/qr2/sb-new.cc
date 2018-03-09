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

complex<float> buffer[1024];
complex<float> _one(1, 0);

void qr(complex<float> *a, complex<float> *tau) {
  int N = _N_;
  const int w_spad = 8;
  const int r_trans_spad = N * 8;

  SB_CONTEXT(2);
  SB_CONFIG(fused1_config, fused1_size);

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
  SB_REPEAT_PORT(N * (N - 1));
  SB_XFER_RIGHT(P_hh_TAUxfer, P_fused1_TAU, 1);
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hh_V);
  SB_XFER_RIGHT(P_hh_O, P_fused1_IN, N - 1);

  SB_CONTEXT(2);
  SB_DMA_WRITE(P_fused1_OUTmem, 8 * N, 8, N - 1, a + N);
  SB_SCR_WRITE(P_fused1_OUTspad, 8 * (N - 1), w_spad);
  SB_DMA_SCRATCH_LOAD(&_one, 8, 8, 1, 0);
  SB_WAIT_SCR_WR();
  SB_SCR_PORT_STREAM(0, 0, 8 * N, N - 1, P_fused1_W);
  SB_2D_CONST(P_fused1_reset, 2, N - 1, 1, 1, N - 1);
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i, 8 * N, 8, N, P_fused1_M);
    SB_CONST(P_fused1_HEAD, *(uint64_t*)(a + i), N);
  }

  SB_DMA_WRITE(P_fused1_FIN, 8, 8, N - 1, a + 1);

  SB_REPEAT_PORT(N - 1);
  SB_RECURRENCE(P_fused1_V, P_fused1_A, N - 1);
  SB_SCR_PORT_STREAM(w_spad, 0, 8 * (N - 1), N - 1, P_fused1_B);
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i + N, 8 * N, 8, N - 1, P_fused1_C);
  }

  for (int i = 1; i < N - 1; ++i) {
    int n = N - i;
    printf(""); SB_RECURRENCE(P_fused1_RES, P_fused1_IN, n);
    printf(""); SB_SCR_WRITE(P_fused1_RES, n * (n - 1) * 8, r_trans_spad);
    printf(""); SB_XFER_LEFT(P_fused1_OUTmem, P_hh_A, n);
    printf(""); SB_XFER_LEFT(P_fused1_OUTspad, P_hh_HEAD, 1);
    printf(""); SB_XFER_LEFT(P_fused1_OUTspad, P_hh_V, n - 1);
    printf(""); SB_CONTEXT(1);
    printf(""); SB_CONST(P_hh_reset, 2, n - 1);
    printf(""); SB_CONST(P_hh_reset, 1, 1);
    printf(""); SB_RECURRENCE(P_hh_NORM2_, P_hh__NORM2, 1);
    printf(""); SB_DMA_WRITE(P_hh_ALPHA, 0, 8, 1, a + i * N + i);
    printf(""); SB_DMA_WRITE(P_hh_TAUwrite, 0, 8, 1, tau + i);
    printf(""); SB_REPEAT_PORT(n * (n - 1));
    printf(""); SB_XFER_RIGHT(P_hh_TAUxfer, P_fused1_TAU, 1);
    printf(""); SB_REPEAT_PORT(n - 1);
    printf(""); SB_RECURRENCE(P_hh_U1inv, P_hh_DIV, 1);
    printf(""); SB_XFER_RIGHT(P_hh_O, P_fused1_IN, n - 1);
    printf(""); SB_CONTEXT(2);
    printf(""); SB_DMA_WRITE(P_fused1_OUTmem, 8 * N, 8, n - 1, a + (i + 1) * N + i);
    printf(""); SB_SCR_WRITE(P_fused1_OUTspad, (n - 1) * 8, w_spad);
    printf(""); SB_WAIT_SCR_WR();
    printf(""); SB_SCR_PORT_STREAM(0, 0, 8 * n, n - 1, P_fused1_W);
    printf(""); SB_2D_CONST(P_fused1_reset, 2, n - 1, 1, 1, n - 1);
    printf(""); SB_SCRATCH_READ(r_trans_spad, n * (n - 1) * 8, P_fused1_M);
    printf(""); SB_REPEAT_PORT(n);
    printf(""); SB_SCR_PORT_STREAM(r_trans_spad, 8 * n, 8, n - 1, P_fused1_HEAD);
    printf(""); SB_DMA_WRITE(P_fused1_FIN, 8, 8, n - 1, a + i * N + i + 1);
    printf(""); SB_REPEAT_PORT(n - 1);
    printf(""); SB_RECURRENCE(P_fused1_V, P_fused1_A, n - 1);
    printf(""); SB_SCR_PORT_STREAM(w_spad, 0, 8 * (n - 1), n - 1, P_fused1_B);
    printf(""); SB_SCR_PORT_STREAM(r_trans_spad + 8, 8 * n, 8 * (n - 1), (n - 1), P_fused1_C);
  }

  SB_DMA_WRITE(P_fused1_RES, 8, 8, 1, a + N * N - 1);

  SB_CONTEXT(3);
  SB_WAIT_ALL();
}

void unitary(complex<float> *a, complex<float> *tau, complex<float> *q) {
  int N = _N_;
  q[N * N - 1] = 1.0f;
  complex<float> *w = buffer;
  complex<float> *v = buffer + N;
  for (int i = N - 2; i >= 0; --i) {
    int n = N - i;
    for (int j = i + 1; j < N; ++j)
      w[j - i - 1] = a[j * N + i];
    SBvec_mul_mat(q + (i + 1) * N + i + 1, n - 1, n - 1, N, w, true, v);

    complex<float> neg(-tau[i]);
    {
      SB_CONFIG(nmlz_config, nmlz_size);
      //nmlz1:
      int pad = get_pad(n - 1, 4);
      SB_DMA_READ(v, 0, 8 * (n - 1), 1, P_nmlz_A); SB_CONST(P_nmlz_A, 0, pad);
      SB_CONST(P_nmlz_B, *((uint64_t*)&neg), n - 1 + pad);
      SB_DMA_WRITE(P_nmlz_AB, 8, 8, n - 1, q + i * N + i + 1); SB_GARBAGE(P_nmlz_AB, pad);
      SB_GARBAGE(P_nmlz_AB_, n - 1 + pad);

      //nmlz2:
      SB_DMA_READ(a + (i + 1) * N + i, 8 * N, 8, n - 1, P_nmlz_A); SB_CONST(P_nmlz_A, 0, pad);
      SB_CONST(P_nmlz_B, *((uint64_t*)&neg), n - 1 + pad);
      SB_DMA_WRITE(P_nmlz_AB, 8 * N, 8, n - 1, q + (i + 1) * N + i); SB_GARBAGE(P_nmlz_AB, pad);
      SB_GARBAGE(P_nmlz_AB_, n - 1 + pad);

      SB_WAIT_ALL();
    }

    {
      int pad = (n - 1) & 1;
      SB_CONFIG(finalize_config, finalize_size);
      SB_CONST(P_finalize_TAU, *((uint64_t*)(tau + i)), (n - 1) * (n - 1 + pad));
      for (int j = 0; j < n - 1; ++j) {
        SB_DMA_READ(q + (j + i + 1) * N + i + 1, 0, 8 * (n - 1), 1, P_finalize_C); SB_CONST(P_finalize_C, 0, pad);
        SB_CONST(P_finalize_B, *((uint64_t*)(w + j)), n - 1); SB_CONST(P_finalize_B, 0, pad);
        SB_DMA_READ(v, 0, 8 * (n - 1), 1, P_finalize_A); SB_CONST(P_finalize_A, 0, pad);
        SB_DMA_WRITE(P_finalize_O, 0, 8 * (n - 1), 1, q + (j + i + 1) * N + i + 1); SB_GARBAGE(P_finalize_O, pad);
      }
      SB_WAIT_ALL();
    }

    q[i * N + i] = 1.0f - tau[i];
    //for (int j = i + 1; j < N; ++j) {
    //  // nmlz1: q[i * N + j] = -tau[i] * v[j - i - 1];
    //  // nmlz2: q[j * N + i] = -tau[i] * a[j * N + i];
    //  for (int k = i + 1; k < N; ++k) {
    //    q[j * N + k] -= tau[i] * w[j - i - 1] * v[k - i - 1];
    //  }
    //}
  }
}
