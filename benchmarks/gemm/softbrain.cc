#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "compute.h"
#include "softbrain-config/fixed_point.h"
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

const complex<int16_t> _zero(0, 0);
union reinterprete_t {
  complex<int16_t> a[2];
  uint64_t v;
} double_zero = {_zero, _zero};

void gemm(int n, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  SB_CONFIG(compute_config, compute_size);
  SB_DMA_READ(b, 8, 8 * n * n / 2, n, P_compute_B);
  for (int i = 0; i < n; ++i) {
    SB_CONST(P_compute_C, double_zero.v, n / 2);
    SB_RECURRENCE(P_compute_O, P_compute_C, n * (n - 1) / 2);
    for (int k = 0; k < n; ++k) {
      complex<int16_t> tmp = a[i * n + k];
      reinterprete_t dup = {tmp, tmp};
      SB_CONST(P_compute_A, dup.v, n / 2);
    }
    SB_DMA_WRITE(P_compute_O, 8, 8, n / 2, c + i * n);
  }
  SB_WAIT_ALL();
}
