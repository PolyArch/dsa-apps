#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "sim_timing.h"

#include <complex>

using std::complex;

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

void filter(complex<float> *a, complex<float> *b, complex<float> *c) {
  for (int i = 0; i < N - FILTER + 1; i += DECIMATION) {
    complex<float> sum(0, 0);
    for (int j = 0; j < FILTER; ++j) {
      complex<float> tmp(complex_mul(a[i + j], b[j]));
      sum = complex<float>(complex_add(sum, tmp));
    }
    c[i / DECIMATION] = sum;
    //std::cout << i / DECIMATION << "\n";
  }
}

