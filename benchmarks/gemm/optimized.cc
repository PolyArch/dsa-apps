#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "softbrain-config/fixed_point.h"

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

void gemm(int n, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; k += 2) {
      complex<int16_t> tmp0 = a[i * n + k];
      complex<int16_t> tmp1 = a[i * n + k + 1];
      for (int j = 0; j < n; j += 2) {
        c[i * N + j] =
          complex<int16_t>(complex_add(c[i * N + j], complex<int16_t>(complex_mul(tmp0, b[k * n + j]))));
        c[i * N + j + 1] =
          complex<int16_t>(complex_add(c[i * N + j + 1], complex<int16_t>(complex_mul(tmp0, b[k * n + j + 1]))));
        c[i * N + j] =
          complex<int16_t>(complex_add(c[i * N + j], complex<int16_t>(complex_mul(tmp1, b[(k + 1) * n + j]))));
        c[i * N + j + 1] =
          complex<int16_t>(complex_add(c[i * N + j + 1], complex<int16_t>(complex_mul(tmp1, b[(k + 1) * n + j + 1]))));
      }
    }
  }
}
