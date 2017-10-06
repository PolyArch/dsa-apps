#include <complex>
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

void transform(int n, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  for (int i = 0; i < n; ++i) {
    c[i] = complex<int16_t>(0, 0);
    complex<int16_t> sum(0, 0);
    for (int j = 0; j < n; ++j) {
      complex<int16_t> tmp(complex_mul(a[i * N + j], b[j]));
      sum = complex<int16_t>(complex_add(sum, tmp));
    }
    c[i] = sum;
  }
}
