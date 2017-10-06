#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>

#define complex_mul(a, b) \
  ((a).real() * (b).real() - (a).imag() * (b).imag()), \
  ((a).real() * (b).imag() + (a).imag() * (b).real())

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

#define PI 3.14159265358979303

void show_matrix(int n, complex<float> *a) {
  std::cout << std::fixed;
  std::cout.precision(5);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      std::cout << a[i * n + j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void gemm(int n, complex<float> *a, complex<float> *b, complex<float> *c) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      complex<float> tmp = a[i * n + k];
      for (int j = 0; j < n; ++j) {
        c[i * n + j] = complex<float>(complex_add(c[i * n + j], complex<float>(complex_mul(tmp, b[k * n + j]))));
      }
    }
  }
}
