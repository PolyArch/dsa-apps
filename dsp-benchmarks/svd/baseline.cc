#include "svd.h"
#include <iostream>
#include <iomanip>
#include "sim_timing.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

complex<float> _tmp[N * N];
const complex<float> _one(1, 0), _zero(0, 0);

void show_matrix(const char *name, complex<float> *a)  {
  std::cout << name << ":\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << a[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

float norm(complex<float> *a, int n) {
  float res = 0;
  for (int i = 0; i < n; ++i) {
    res += (a[i] * std::conj(a[i])).real();
  }
  return sqrt(res);
}

complex<float> at_a[N * N];

void svd(complex<float> *a, complex<float> *u, complex<float> *s, complex<float> *v) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex<float> sum(0, 0);
      for (int k = 0; k < N; ++k) {
        complex<float> tmp(complex_conj_mul(a[k * N + i], a[k * N + j]));
        sum = complex<float>(complex_add(sum, tmp));
      }
      at_a[i * N + j] = sum;
    }
  }

  for (int i = 1; i < N; ++i) {
    int len = N - i;
    complex<float> v[len];
    v[0] = at_a[(i - 1) * N + i];
    float norm0 = complex_norm(v[0]), norm1 = 0.;
    for (int j = i + 1; j < N; ++j) {
      v[j - i] = at_a[i * N + j];
      norm1 += complex_norm(v[j - i]);
    }
    float rate = sqrt(1 + norm1 / norm0) + 1;
    v[0] = complex<float>(v[0].real() * rate, v[0].imag() * rate);
    norm1 = 1 / sqrt(complex_norm(v[0]) + norm1);
    for (int j = 0; j < len; ++j) {
      v[j] = complex<float>(v[j].real() * norm1, v[j].imag() * -norm1);
      std::cout << v[j] << " ";
    }
    std::cout << "\n";
  }
}

