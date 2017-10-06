#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "sb_insts.h"
#include "compute.h"

#define complex_mul(a, b) \
  ((a).real() * (b).real() - (a).imag() * (b).imag()), \
  ((a).real() * (b).imag() + (a).imag() * (b).real())

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

const complex<float> _zero(0, 0);

#define PI 3.14159265358979303

void gemm(int n, complex<float> *a, complex<float> *b, complex<float> *c) {
  SB_CONFIG(compute_config, compute_size);
  SB_DMA_READ(b, 8, 8 * n * n, n, P_compute_B);
  for (int i = 0; i < n; ++i) {
    SB_CONST(P_compute_C, *((uint64_t*)&_zero), n);
    SB_RECURRENCE(P_compute_O, P_compute_C, n * (n - 1));
    for (int k = 0; k < n; ++k) {
      complex<float> tmp = a[i * n + k];
      SB_CONST(P_compute_A, *((uint64_t*) &tmp), n);
    }
    SB_DMA_WRITE(P_compute_O, 8, 8, n, c + i * n);
  }
  SB_WAIT_ALL();
}
