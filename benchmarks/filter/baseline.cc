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


void filter(complex<float> *a, complex<float> *b, complex<float> *c) {
  for (int i = 0; i < N - FILTER + 1; ++i) {
    c[i] = complex<float>(0, 0);
    for (int j = 0; j < FILTER; ++j) {
      c[i] += a[i + j] * b[j];
    }
  }
}

