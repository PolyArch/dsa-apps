#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include <complex>

using std::complex;
const complex<float> _zero(0, 0); 
#include <complex>
#include <cmath>
#include <algorithm>
#include "sb_insts.h"

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

void fft(complex<float> *_a, complex<float> *w, int n, bool conj) {
  complex<float> *_buffer = new complex<float>[n];
  complex<float> *from = _a, *to = _buffer;
  for (int blocks = n / 2; blocks; blocks >>= 1) {
    int span = n / blocks;
    for (int j = 0; j < span / 2 * blocks; j += blocks) {
      complex<float> coef = w[j];
      for (int i = 0; i < blocks; ++i) {
        complex<float> &L = from[2 * j + i];
        complex<float> &R = from[2 * j + i + blocks];
        complex<float> tmp;
        tmp = conj ? complex<float>(complex_conj_mul(coef, R)) : complex<float>(complex_mul(coef, R));
        to[i + j] = complex<float>(complex_add(L, tmp));
        to[i + j + span / 2 * blocks] = complex<float>(complex_sub(L, tmp));
      }
    }
    swap(from, to);
  }
  if (from != _a) {
    for (int i = 0; i < n; ++i) {
      _a[i] = _buffer[i];
    }
  }
}

//static complex<float> a[N + M << 1];
//static complex<float> b[N + M << 1];

void filter(
    int n, int m,
    complex<float> *_a,
    complex<float> *_b,
    complex<float> *c,
    complex<float> *w
) {
  int _n;
  for (_n = 1; _n < n + m; _n <<= 1);
  complex<float> *a = new complex<float>[_n];
  complex<float> *b = new complex<float>[_n];
  for (int i = 0; i < _n; ++i) {
    a[i] = i < n ? _a[i] : _zero;
  }
  for (int i = 0; i < _n; ++i) {
    b[i] = i >= n && i < n + m ? _b[m - 1 - (i - n)] : _zero;
  }
  fft(a, w, _n, false);
  fft(b, w, _n, false);
  for (int i = 0; i < _n; ++i)
    a[i] = complex<float>(complex_mul(a[i], b[i]));
  fft(a, w, _n, true);
  float n1 = 1. / (float) _n;
  //for (int i = 0; i < _n; ++i) {
    //std::cout << a[i] * n1 << "\n";
  //}
  //std::cout << "\n";
  for (int i = 0, j = n + m - 1; i < n - m + 1; ++i, j = (j + 1) & (_n - 1))
    c[i] = a[j] * n1;
}

