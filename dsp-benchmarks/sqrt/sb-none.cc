#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "matvec.h"

using std::complex;

void sqrt(complex<float> *a) {
  for (int i = 0; i < N; ++i) {
    a[i] = sqrt(a[i].real());
  }
}

