#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "sim_timing.h"

#define PI 3.14159265358979303

#include <complex>

using std::complex;
const complex<float> _zero(0, 0);

#include <complex>
#include <cmath>
#include <algorithm>
#include "sb_insts.h"
#include "compute.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;
#define PI 3.14159265358979303

complex<float> _one = complex<float>(1, 0);

void fft(complex<float> *_a, complex<float> *w, bool inverse, int n) {
  SB_CONFIG(compute_config, compute_size);
  complex<float> *_buffer = new complex<float>[n];
  complex<float> *from = _a, *to = _buffer;
  for (int blocks = n / 2; blocks; blocks >>= 1) {
    int span = n / blocks;
    SB_DMA_READ(from,          2 * blocks * 8, blocks * 8, span / 2, P_compute_L)
    SB_DMA_READ(from + blocks, 2 * blocks * 8, blocks * 8, span / 2, P_compute_R)
    for (int j = 0; j < span / 2; ++j) {
      complex<float> coef = w[j * blocks];
      if (inverse) {
        coef = _one / coef;
      }
      SB_CONST(P_compute_W, *((unsigned long long*) &coef), blocks);
      /*for (int i = 0; i < blocks; ++i) {
        //printf("%d %d %d\n", blocks, j, i);
        complex<float> &L = from[2 * j + i];
        complex<float> &R = from[2 * j + i + blocks];
        complex<float> tmp(complex_mul(w[j], R));
        to[i + j] = complex<float>(complex_add(L, tmp));
        to[i + j + span / 2 * blocks] = complex<float>(complex_sub(L, tmp));
      }*/
    }
    SB_DMA_WRITE(P_compute_A, 8, 8, n / 2, to);
    SB_DMA_WRITE(P_compute_B, 8, 8, n / 2, to + n / 2);

    SB_WAIT_ALL();
    swap(from, to);
  }
  if (from != _a) {
    for (int i = 0; i < n; ++i) {
      _a[i] = _buffer[i];
    }
  }
}



void filter(complex<float> *_a, complex<float> *_b, complex<float> *c) {
  std::cout << std::fixed;
  int n;
  for (n = 1; n < N + FILTER;n <<= 1);
  complex<float> *a = new complex<float>[n];
  complex<float> *b = new complex<float>[n];
  complex<float> *w = new complex<float>[n / 2];
  for (int i = 0; i < n; ++i) {
    a[i] = i < N ? _a[i] : _zero;
  }
  for (int i = 0; i < n; ++i) {
    b[i] = i >= N && i < N + FILTER ? _b[FILTER - 1 - (i - N)] : _zero;
  }
  end_roi();
  for (int i = 0; i < n / 2; ++i) {
    w[i] = complex<float>(cos(2 * PI * i / n), sin(2 * PI * i / n));
  }
  begin_roi();
  fft(a, w, false, n);
  fft(b, w, false, n);
  for (int i = 0; i < n; ++i) {
    a[i] *= b[i];
  }
  fft(a, w, true, n);
  //for (int i = 0; i < n; ++i) std::cout << a[i] / (float)n << " "; std::cout << "\n";
  for (int i = 0; i < N - FILTER + 1; ++i) {
    c[i] = a[n - 1 - FILTER + i] / (float)(n);
  }
}

