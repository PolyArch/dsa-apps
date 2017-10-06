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


void transform(int n, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  SB_CONFIG(compute_config, compute_size);
  SB_DMA_READ(a, 0, n * n * 4, 1, P_compute_A);
  SB_DMA_READ(b, 0, n * 4, n, P_compute_B);
  for (int i = 0; i < n; i += 2) {
    SB_CONST(P_compute_reset, 0, N / 4 - 1);
    SB_CONST(P_compute_reset, 1, 1);
    SB_GARBAGE(P_compute_SUM, N / 4 - 1);
    SB_DMA_WRITE(P_compute_SUM, 0, 8, 1, c + i);
  }
}
