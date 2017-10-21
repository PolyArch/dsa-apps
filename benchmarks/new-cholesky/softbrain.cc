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
    div= std::sqrt(*a);
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
    {
      L[0] = div;
      float norm = div.real() * div.real() + div.imag() * div.imag();
      norm = 1 / norm;
      for (int i = 1; i < N; ++i) {
        complex<float> tmp(complex_mul(div, a[i]));
        L[i * N] = complex<float>(tmp.real() * norm, tmp.imag() * norm);
      }
    }
  }
  for (int i = 1; i < N; ++i) {
    SB_GARBAGE(P_compute_O1, N - i);
    SB_RECURRENCE(P_compute_O1, P_compute_Z, (N - i - 1) * (N - i) / 2);
    complex<float> tmp;
    SB_RECV(P_compute_O2, tmp);
    float norm = 1 / (tmp.real() * tmp.real() + tmp.imag() * tmp.imag());
    union {
      float f[2];
      uint64_t v;
    } ri_norm = {-norm, -norm}, ri_v = {tmp.real(), tmp.imag()};
    a[i * (N + 1)] = tmp;
    div = tmp;
    for (int j = i + 1; j < N; ++j) {
      SB_RECV(P_compute_O2, tmp);
      a[i * N + j] = tmp;
    }
    SB_GARBAGE(P_compute_O2, (N - i - 1) * (N - i) / 2);
    for (int j = i + 1; j < N; ++j) {
      int len = N - j;
      SB_DMA_READ(a + i * N + j, 0, 8, len, P_compute_A);
      SB_DMA_READ(a + i * N + j, 0, 8 * len, 1, P_compute_B);
      SB_CONST(P_compute_NORM, ri_norm.v, len);
      SB_CONST(P_compute_V, ri_v.v, len);
    }
    {
      div = std::sqrt(div);
      L[i * (N + 1)] = div;
      float norm = div.real() * div.real() + div.imag() * div.imag();
      norm = 1 / norm;
      for (int j = i + 1; j < N; ++j) {
        complex<float> tmp(complex_mul(div, a[i * N + j]));
        L[j * N + i] = complex<float>(tmp.real() * norm, tmp.imag() * norm);
      }
    }
  }
  SB_WAIT_ALL();
}

