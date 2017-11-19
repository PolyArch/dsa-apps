#include <complex>
#include <cmath>
#include <algorithm>

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

complex<float> _buffer[N];

void fft(complex<float> *_a, complex<float> *w) {
  complex<float> *from = _a, *to = _buffer;
  for (int blocks = N / 2; blocks; blocks >>= 1) {
    int span = N / blocks;
    for (int j = 0; j < span / 2 * blocks; j += blocks) {
      for (int i = 0; i < blocks; ++i) {
        //printf("%d %d %d\n", blocks, j, i);
        complex<float> &L = from[2 * j + i];
        complex<float> &R = from[2 * j + i + blocks];
        complex<float> tmp(complex_mul(w[j], R));
        to[i + j] = complex<float>(complex_add(L, tmp));
        to[i + j + span / 2 * blocks] = complex<float>(complex_sub(L, tmp));
      }
    }
    swap(from, to);
  }
  if (from != _a) {
    for (int i = 0; i < N; ++i) {
      _a[i] = _buffer[i];
    }
  }
}
