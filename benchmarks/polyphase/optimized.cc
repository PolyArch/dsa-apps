#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "sim_timing.h"

#define PI 3.14159265358979303

#include <complex>

using std::complex;
const complex<float> _zero(0, 0);

#include <complex>
#include <algorithm>

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;
#define PI 3.14159265358979303

#define L ((FILTER - 1) / DECIMATION + 1)
#define D DECIMATION

complex<float> _buf[D][L];
complex<float> _sum[D];
int _idx[D];

void filter(complex<float> *a, complex<float> *b, complex<float> *c) {
  std::cout << std::fixed;
  int _filter = L * D;
  for (int i = 0; i < N; ++i) {
    int bin = i % D;
    int &idx = _idx[bin];
    _buf[bin][idx] = a[i];
    _sum[bin] = complex<float>(0, 0);
    for (int j = 0; j < L; ++j) {
      if (bin + j * D < FILTER) {
        _sum[bin] += _buf[bin][(idx + j + 1) % L] * b[bin + j * D];
        //std::cout << _buf[bin][(idx + j + 1) % L] << " x " << b[bin + j * D] << "\n";
      }
    }
    idx = (idx + 1) % L;
    complex<float> sum(0, 0);
    if ((i - _filter + 1) >= 0 && ((i - _filter + 1)) % DECIMATION == 0) {
      //std::cout << (i - _filter + 1) % DECIMATION << "\n";
      for (int j = 0; j < D; ++j) {
        //std::cout << _sum[j] << "\n";
        sum += _sum[j];
      }
      //std::cout << sum << "\n\n";
      c[(i - _filter + 1) / DECIMATION] = sum;
    }
  }
}

