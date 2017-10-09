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
complex<int16_t> _zero(0, 0);

void gemm(int n, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  SB_CONFIG(compute_config, compute_size);
  for (int i = 0; i < n; ++i) {
    SB_DMA_READ(b    , 8 * n, 4 * n, n / 2, P_compute_B0);
    SB_DMA_READ(b + n, 8 * n, 4 * n, n / 2, P_compute_B1);
    SB_CONST(P_compute_C, *((uint64_t*) &_zero), n / 2);
    SB_RECURRENCE(P_compute_O, P_compute_C, (n / 2) * (n / 2 - 1));
    SB_DMA_WRITE(P_compute_O, 8, 8, n / 2, c + i * N);
    for (int k = 0; k < n; k += 2) {
      SB_DMA_READ(a + i * n + k, 0, 8, n / 2, P_compute_A);
    }
  }
  SB_WAIT_ALL();
}
