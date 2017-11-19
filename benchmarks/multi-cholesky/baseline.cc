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


void cholesky(complex<float> *a, complex<float> *L) {
  complex<float> *A = new complex<float>[N * N];
  for (int i = 0; i < N * N; ++i) {
    A[i] = a[i];
  }
  for (int i = 0; i < N; ++i) {
    complex<float> div = std::sqrt(A[i * (N + 1)]);
    complex<float> *b = A + i * (N + 1);
    L[i * (N + 1)] = div;
    for (int j = i + 1; j < N; ++j) {
      L[j * N + i] = b[j - i] / div;
    }
    for (int j = i + 1; j < N; ++j) {
      for (int k = i + 1; k < N; ++k) {
        A[j * N + k] -= std::conj(b[j - i]) * b[k - i] / A[i * (N + 1)];
      }
    }
    for (int j = i + 1; j < N; ++j)
      A[i * N + j] = A[j * N + i] = 0;
    A[i * (N + 1)] = 1;
  }
  delete []A;
}

