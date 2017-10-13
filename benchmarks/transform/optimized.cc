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
  for (int i = 0; i < n; i += 2) {
    c[i] = complex<int16_t>(0, 0);
    c[i + 1] = complex<int16_t>(0, 0);
    complex<int16_t> sum0(0, 0);
    complex<int16_t> sum1(0, 0);
    for (int j = 0; j < n; j += 2) {
      complex<int16_t> tmp00(complex_mul(a[i * N + j], b[j]));
      complex<int16_t> tmp01(complex_mul(a[i * N + j + 1], b[j + 1]));
      complex<int16_t> tmp10(complex_mul(a[(i + 1) * N + j], b[j]));
      complex<int16_t> tmp11(complex_mul(a[(i + 1) * N + j + 1], b[j + 1]));
      sum0 = complex<int16_t>(complex_add(sum0, tmp00));
      sum0 = complex<int16_t>(complex_add(sum0, tmp01));
      sum1 = complex<int16_t>(complex_add(sum1, tmp10));
      sum1 = complex<int16_t>(complex_add(sum1, tmp11));
    }
    c[i] = sum0;
    c[i + 1] = sum1;
  }
}
