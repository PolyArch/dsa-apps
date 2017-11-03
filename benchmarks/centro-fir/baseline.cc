#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "softbrain-config/fixed_point.h"

#define complex_mul(a, b) \
  (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

#define PI 3.14159265358979303

void filter(int n, int m, complex<float> *a, complex<float> *b, complex<float> *c) {
  for (int i = 0; i < n - m + 1; ++i) {
    complex<float> sum(0, 0);
    for (int j = 0; j < m; ++j)
      sum =
        complex<float>(complex_add(sum, complex<float>(complex_mul(a[i + j], b[j]))));
    c[i] = sum;
  }
}
