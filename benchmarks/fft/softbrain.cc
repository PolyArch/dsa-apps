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
complex<float> _buffer[N];
#define PI 3.14159265358979303

void fft(complex<float> *_a, complex<float> *w) {
  SB_CONFIG(compute_config, compute_size);
  complex<float> *from = _a, *to = _buffer;
  for (int blocks = N / 2; blocks; blocks >>= 1) {
    int span = N / blocks;
    SB_DMA_READ(from,          2 * blocks * 8, blocks * 8, span / 2, P_compute_L)
    SB_DMA_READ(from + blocks, 2 * blocks * 8, blocks * 8, span / 2, P_compute_R)
    for (int j = 0; j < span / 2; ++j) {
      SB_CONST(P_compute_W, *((unsigned long long*)(w + j * blocks)), blocks);
      /*for (int i = 0; i < blocks; ++i) {
        //printf("%d %d %d\n", blocks, j, i);
        complex<float> &L = from[2 * j + i];
        complex<float> &R = from[2 * j + i + blocks];
        complex<float> tmp(complex_mul(w[j], R));
        to[i + j] = complex<float>(complex_add(L, tmp));
        to[i + j + span / 2 * blocks] = complex<float>(complex_sub(L, tmp));
      }*/
    }
    SB_DMA_WRITE(P_compute_A, 8, 8, N / 2, to);
    SB_DMA_WRITE(P_compute_B, 8, 8, N / 2, to + N / 2);

    SB_WAIT_ALL();
    swap(from, to);
  }
  if (from != _a) {
    for (int i = 0; i < N; ++i) {
      _a[i] = _buffer[i];
    }
  }
}
