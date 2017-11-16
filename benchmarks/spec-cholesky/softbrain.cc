#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "compute.h"
#include "offload.h"
#include "writeback.h"
#include "sb_insts.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

void cholesky(complex<float> *a, complex<float> *L) {
  //SB_CONTEXT(1);
  //SB_CONFIG(compute_config, compute_size);
  SB_CONTEXT(2);
  SB_CONFIG(offload_config, offload_size);
  SB_CONTEXT(4);
  SB_CONFIG(writeback_config, writeback_size);
  complex<float> div;
  {
    SB_CONTEXT(2);
    SB_DMA_READ(a, 0, 8 * N, 1, P_offload_V);
    SB_CONST(P_offload_MUX, 0, 1);
    SB_CONST(P_offload_MUX, 1, N - 1);

    //SB_REPEAT_PORT(N * (N - 1) / 2);
    //SB_XFER_LEFT(P_offload_INV, P_compute_V, 1);
    SB_GARBAGE(P_offload_INV, 1);
    SB_RECV(P_offload_SQRT, L[0]);
    SB_REPEAT_PORT(N - 1);
    SB_XFER_RIGHT(P_offload_SQRTINV, P_writeback_DIV, 1);

    SB_XFER_RIGHT(P_offload_STREAM, P_writeback_BP, N - 1);
    //SB_GARBAGE(P_offload_STREAM, N - 1);

    /*SB_CONTEXT(1);
    complex<float> *bj = a + 1, *bk, v = *a;
    SB_DMA_READ_STRETCH(a + N + 1, 8, 8 * N, -8, N - 1, P_compute_Z);
    SB_DMA_READ_STRETCH(a + 1, 8, 8 * N, -8, N - 1, P_compute_B);
    for (int j = 1; j < N; ++j)
      SB_CONST(P_compute_A, *((uint64_t*)a + j), N - j);*/

    SB_CONTEXT(4);
    SB_DMA_WRITE(P_writeback_RES, N * 8, 8, N - 1, L + N);
  }
  SB_CONTEXT(7);
  SB_WAIT_ALL();
  /*int addr = (N - 1) * 8;
  for (int i = 1; i < N; ++i) {
    SB_CONTEXT(1);
    complex<float> tmp0, tmp1;
    SB_RECV(P_compute_O0, tmp0);
    SB_GARBAGE(P_compute_O1, 1);
    SB_SCR_WRITE(P_compute_O0, (N - i - 1) * 8, addr);
    SB_XFER_RIGHT(P_compute_O1, P_writeback_BP, N - i - 1);
    SB_RECURRENCE(P_compute_O0, P_compute_Z, (N - i - 1) * (N - i) / 2);
    SB_GARBAGE(P_compute_O1, (N - i - 1) * (N - i) / 2);
    float norm = 1 / (tmp0.real() * tmp0.real() + tmp0.imag() * tmp0.imag());
    union {
      float f[2];
      uint64_t v;
    } ri_norm = {-norm, -norm}, ri_v = {tmp0.real(), tmp0.imag()};
    tmp1 = L[i * (N + 1)] = std::sqrt(tmp0);

    int total = N - i - 1;
    SB_CONST(P_compute_NORM, ri_norm.v, (1 + total) * total / 2);
    SB_CONST(P_compute_V, ri_v.v, (1 + total) * total / 2);

    SB_WAIT_SCR_WR();

    SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * (N - i - 1), -8, N - i - 1, P_compute_B);
    for (int j = i + 1, cur = addr; j < N; ++j) {
      int len = N - j;
      SB_REPEAT_PORT(len);
      SB_SCR_PORT_STREAM(cur, 0, 8, 1, P_compute_A);
      //SB_SCR_PORT_STREAM(cur, 0, 8 * len, 1, P_compute_B);
      cur += 8;
    }
    addr += (N - i - 1) * 8;

    SB_CONTEXT(2);
    ri_norm.f[0] = ri_norm.f[1] = 1. / (tmp1.real() * tmp1.real() + tmp1.imag() * tmp1.imag());
    SB_CONST(P_writeback_NORM, ri_norm.v, N - i - 1);
    SB_CONST(P_writeback_DIV, *((uint64_t *) &tmp1), N - i - 1);
    SB_DMA_WRITE(P_writeback_RES, N * 8, 8, N - i - 1, L + (i + 1) * N + i);

  }
  SB_CONTEXT(3);
  SB_WAIT_ALL();*/
}

