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
#include "writeback.h"
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
  SB_CONFIG(compute_config, compute_size);
  complex<float> div;
  {
    complex<float> *bj = a + 1, *bk, v = *a;
    float norm = 1 / (v.real() * v.real() + v.imag() * v.imag());
    union {
      float f[2];
      uint64_t v;
    } ri_norm = {-norm, -norm}, ri_v = {v.real(), v.imag()};
    for (int j = 1; j < N; ++j) {
      int len = N - j;
      uint64_t ri_bj = *((uint64_t*)bj);
      SB_DMA_READ(a + j * (N + 1), 0, 8 * len, 1, P_compute_Z);
      SB_CONST(P_compute_A, ri_bj, len);
      SB_DMA_READ(bj, 0, 8 * len, 1, P_compute_B);
      SB_CONST(P_compute_NORM, ri_norm.v, len);
      SB_CONST(P_compute_V, ri_v.v, len);
      ++bj;
    }
    *L = sqrt(*a);
  }
  int addr = (N - 1) * 8;
  SB_DMA_SCRATCH_LOAD(a + 1, 0, 8 * (N - 1), 1, 0);
  for (int i = 1; i < N; ++i) {
    complex<float> tmp;
    SB_RECV(P_compute_O, tmp);
    SB_SCR_WRITE(P_compute_O, (N - i - 1) * 8, addr);
    SB_RECURRENCE(P_compute_O, P_compute_Z, (N - i - 1) * (N - i) / 2);
    float norm = 1 / (tmp.real() * tmp.real() + tmp.imag() * tmp.imag());
    union {
      float f[2];
      uint64_t v;
    } ri_norm = {-norm, -norm}, ri_v = {tmp.real(), tmp.imag()};
    L[i * N + i] = std::sqrt(tmp);
    SB_WAIT_SCR_WR();
    int total = N - i - 1;
    SB_CONST(P_compute_NORM, ri_norm.v, (1 + total) * total / 2);
    SB_CONST(P_compute_V, ri_v.v, (1 + total) * total / 2);
    //SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * total, -8, total, P_compute_B);
    for (int j = i + 1, cur = addr; j < N; ++j) {
      int len = N - j;
      SB_REPEAT_PORT(len);
      SB_SCR_PORT_STREAM(cur, 0, 8, 1, P_compute_A);
      SB_SCR_PORT_STREAM(cur, 0, 8 * len, 1, P_compute_B);
      cur += 8;
      //SB_DMA_READ(a + i * N + j, 0, 8 * len, 1, P_compute_B);
    }
    addr += (N - i - 1) * 8;
  }
  SB_WAIT_ALL();
  SB_CONFIG(writeback_config, writeback_size);
  SB_SCR_PORT_STREAM(0, 0, addr, 1, P_writeback_BP);
  for (int i = 0; i < N; ++i) {
    complex<float> div = L[i * N + i];
    float norm = div.real() * div.real() + div.imag() * div.imag();
    norm = 1 / norm;
    union {
      float f[2];
      uint64_t v;
    } ri_norm = {norm, norm};
    //SB_DMA_READ(a + i * N + i + 1, 8, 8, N - i - 1, P_writeback_BP);
    SB_CONST(P_writeback_NORM, ri_norm.v, N - i - 1);
    SB_CONST(P_writeback_DIV, *((uint64_t *) &div), N - i - 1);
    SB_DMA_WRITE(P_writeback_RES, N * 8, 8, N - i - 1, L + (i + 1) * N + i);
  }
  SB_WAIT_ALL();
}

