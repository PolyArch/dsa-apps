#ifndef svd_h
#define svd_h

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex>

#define BN 4
#define NN (N + BN - 1)

#define eps 1e-3
using std::complex;

struct complex_t {
  float real;
  float imag;
};

void svd(complex<float> *, complex<float> *, complex<float> *, complex<float> *);

#endif
