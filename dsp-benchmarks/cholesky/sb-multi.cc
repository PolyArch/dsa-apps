#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "compute_dual.h"
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
  SB_CONTEXT(1);
  SB_CONFIG(compute_dual_config, compute_dual_size);
  SB_CONTEXT(2);
  SB_CONFIG(writeback_config, writeback_size);
  complex<float> div;
  {
    SB_CONTEXT(1);
    complex<float> *bj = a + 1, *bk, v = *a;
    float norm = 1 / (v.real() * v.real() + v.imag() * v.imag());
    complex<float> inv_a(v.real() * norm, -v.imag() * norm);
    for (int j = 1; j < N; ++j) {
      int len = N - j;
      uint64_t ri_bj = *((uint64_t*)bj);
      SB_DMA_READ(a + j * (N + 1), 0, 8 * len, 1, P_compute_dual_Z);
      SB_CONST(P_compute_dual_A, ri_bj, len);
      SB_DMA_READ(bj, 0, 8 * len, 1, P_compute_dual_B);
      SB_CONST(P_compute_dual_V, *((uint64_t *) &inv_a), len);
      ++bj;
    }
    *L = sqrt(v);
    norm = 1. / (L->real() * L->real() + L->imag() * L->imag());
    complex<float> invL(L->real() * norm, -L->imag() * norm);
    SB_CONTEXT(2);
    SB_DMA_READ(a + 1, 8, 8, N - 1, P_writeback_BP);
    SB_CONST(P_writeback_DIV, *((uint64_t *) &invL), N - 1);
    SB_DMA_WRITE(P_writeback_RES, N * 8, 8, N - 1, L + N);
    //for (int i = 0; i < N; ++i) {
    //  printf("L: %f %f\n", L[i * N].real(), L[i * N].imag());
    //  if (i)
    //    std::cout << a[i] / L[0] << "\n";
    //}
    //SB_WAIT_ALL();
  }
  int addr = (N - 1) * 8;
  for (int i = 1; i < N; ++i) {
    SB_CONTEXT(1);
    complex<float> tmp0, tmp1;
    SB_RECV(P_compute_dual_O0, tmp0);
    SB_GARBAGE(P_compute_dual_O1, 1);
    SB_SCR_WRITE(P_compute_dual_O0, (N - i - 1) * 8, addr);
    SB_XFER_RIGHT(P_compute_dual_O1, P_writeback_BP, N - i - 1);
    SB_RECURRENCE(P_compute_dual_O0, P_compute_dual_Z, (N - i - 1) * (N - i) / 2);
    SB_GARBAGE(P_compute_dual_O1, (N - i - 1) * (N - i) / 2);
    float norm = 1 / (tmp0.real() * tmp0.real() + tmp0.imag() * tmp0.imag());
    tmp1 = L[i * (N + 1)] = std::sqrt(tmp0);
    tmp0 = complex<float>(tmp0.real() * norm, -tmp0.imag() * norm);

    int total = N - i - 1;
    SB_CONST(P_compute_dual_V, *((uint64_t*) &tmp0), (1 + total) * total / 2);

    SB_WAIT_SCR_WR();

    SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * (N - i - 1), -8, N - i - 1, P_compute_dual_B);
    for (int j = i + 1, cur = addr; j < N; ++j) {
      int len = N - j;
      SB_REPEAT_PORT(len);
      SB_SCR_PORT_STREAM(cur, 0, 8, 1, P_compute_dual_A);
      //SB_SCR_PORT_STREAM(cur, 0, 8 * len, 1, P_compute_dual_B);
      cur += 8;
    }
    addr += (N - i - 1) * 8;

    SB_CONTEXT(2);
    norm = 1 / (tmp1.real() * tmp1.real() + tmp1.imag() * tmp1.imag());
    tmp1 = complex<float>(tmp1.real() * norm, -tmp1.imag() * norm);
    SB_CONST(P_writeback_DIV, *((uint64_t *) &tmp1), N - i - 1);
    SB_DMA_WRITE(P_writeback_RES, N * 8, 8, N - i - 1, L + (i + 1) * N + i);

  }
  SB_CONTEXT(3);
  SB_WAIT_ALL();
}

