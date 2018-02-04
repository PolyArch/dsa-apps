#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>

#include "cholesky.h"

#include "compute_dual.dfg.h"
#include "finalize.dfg.h"

#include "sb_insts.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

struct complex_t {
  float real, imag;
};

void cholesky(complex<float> *a, complex<float> *L) {
  SB_CONFIG(compute_dual_config, compute_dual_size);
  complex<float> div;
  {
    complex<float> *bj = a + 1, *bk, v = *a;
    float norm = 1 / (v.real() * v.real() + v.imag() * v.imag());
    union {
      float f[2];
      uint64_t v;
    } ri_v = {v.real() * norm, -v.imag() * norm};
    //SB_CONFIG_PORT(N - 1, -1);
    //SB_DMA_READ(a + 1, 8, 8, N - 1, P_compute_dual_A);
    for (int j = 1; j < N; ++j) {
      int len = N - j;
      uint64_t ri_bj = *((uint64_t*)bj);
      SB_DMA_READ(a + j * (N + 1), 0, 8 * len, 1, P_compute_dual_Z);
      SB_CONST(P_compute_dual_A, ri_bj, len);
      SB_DMA_READ(bj, 0, 8 * len, 1, P_compute_dual_B);
      SB_CONST(P_compute_dual_V, ri_v.v, len);
      ++bj;
    }
  }
  for (int i = 1; i < N; ++i) {
    SB_GARBAGE(P_compute_dual_O0, N - i);
    SB_RECURRENCE(P_compute_dual_O0, P_compute_dual_Z, (N - i - 1) * (N - i) / 2);
    complex<float> tmp;
    SB_RECV(P_compute_dual_O1, tmp);
    a[i * N + i] = tmp;
    float norm = 1 / (tmp.real() * tmp.real() + tmp.imag() * tmp.imag());
    union {
      float f[2];
      uint64_t v;
    } ri_v = {tmp.real() * norm, tmp.imag() * -norm};
    for (int j = i + 1; j < N; ++j) {
      SB_RECV(P_compute_dual_O1, tmp);
      a[i * N + j] = tmp;
    }
    SB_GARBAGE(P_compute_dual_O1, (N - i - 1) * (N - i) / 2);
    int len = N - i - 1;
    SB_CONST(P_compute_dual_V, ri_v.v, (len + 1) * len / 2);
    SB_DMA_READ_STRETCH(a + i * (N + 1) + 1, 8, 8 * (N - i - 1), -8, N - i - 1, P_compute_dual_B);
    SB_CONFIG_PORT(N - i - 1, -1);
    SB_DMA_READ(a + i * N + i + 1, 8, 8, N - i - 1, P_compute_dual_A);
    //for (int j = i + 1; j < N; ++j) {
      //SB_DMA_READ(a + i * N + j, 0, 8, (N - j), P_compute_dual_A);
      //SB_DMA_READ(a + i * N + j, 0, 8 * (N - j), 1, P_compute_dual_B);
    //}
  }
  SB_WAIT_ALL();

  SB_CONFIG(finalize_config, finalize_size);
  for (int i = 0; i < N; ++i) {
    SB_CONST(P_finalize_DIV, 0, 1);
    SB_DMA_READ(a + i * (N + 1), 8, 8, N - i, P_finalize_A);
    SB_CONST(P_finalize_sqrt, 1, 1);
    SB_CONST(P_finalize_sqrt, 0, N - i - 1);
    SB_REPEAT_PORT(N - i - 1);
    SB_RECURRENCE(P_finalize_INV, P_finalize_DIV, 1);
    SB_DMA_WRITE(P_finalize_SQRT, 8, 8, 1, L + i * (N + 1));
    SB_DMA_WRITE(P_finalize_RES, N * 8, 8, N - i - 1, L + (i + 1) * N + i);
  }
  SB_WAIT_ALL();
}

