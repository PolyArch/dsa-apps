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
  int mid = m >> 1;
  for (int j0 = 0, j1 = m - 1; j0 != j1; ++j0, --j1) {
    for (int i = 0; i < n - m + 1; ++i) {
      complex<float> tmp(complex<float>(complex_add(a[i + j0], a[i + j1])));
      c[i] =
        complex<float>(complex_add(c[i], complex<float>(complex_mul(tmp, b[j0]))));
    }
    //c[i] = sum;
  }
  for (int i = 0; i < n - m + 1; ++i) {
    std::cout << c[i] << "\n";
    c[i] = complex<float>(complex_add(c[i], complex<float>(complex_mul(a[i + mid], b[mid]))));
  }
}
