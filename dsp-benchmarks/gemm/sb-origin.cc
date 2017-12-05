#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "softbrain-config/fixed_point.h"
#include "compute.h"
#include "sb_insts.h"

#define complex_mul(a, b) \
  FIX_MINUS(FIX_MUL((a).real(), (b).real()), FIX_MUL((a).imag(), (b).imag())), \
  FIX_ADD(FIX_MUL((a).real(), (b).imag()), FIX_MUL((a).imag(), (b).real()))

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) FIX_ADD((a).real(), (b).real()), FIX_ADD((a).imag(), (b).imag())

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

#define PI 3.14159265358979303

void gemm(int n, int m, int p, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  SB_CONFIG(compute_config, compute_size);
  for (int i = 0; i < n; ++i) {
    SB_CONST(P_compute_C, 0, p / 2);
    SB_RECURRENCE(P_compute_O, P_compute_C, (p / 2) * (m / 2 - 1));
    SB_DMA_READ(b    , 8 * p, 4 * p, m / 2, P_compute_BE);
    SB_DMA_READ(b + p, 8 * p, 4 * p, m / 2, P_compute_BO);
    SB_DMA_WRITE(P_compute_O, 0, 4 * p, 1, c + i * p);
    for (int k = 0; k < m; k += 2) {
      SB_DMA_READ(a + i * m + k, 0, 8, p / 4, P_compute_A);
    }
  }
  SB_WAIT_ALL();
}
