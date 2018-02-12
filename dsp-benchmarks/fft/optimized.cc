#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "sim_timing.h"

using std::complex;

void fft(complex<float> *a, complex<float> *w) {
  int N = _N_;
  for (int span = N >> 1, _log = 0; span; span >>= 1, ++_log) {
    for (int odd = span, even; odd < N; ++odd) {
      odd |= span;
      even = odd ^ span;

      complex<float> temp = a[even] + a[odd];
      a[odd]  = a[even] - a[odd];
      a[even] = temp;

      int index = (even << _log) & (N - 1);
      if (index) {
        a[odd] *= w[index];
        //printf("[%d] %d\n", span, index);
      }
    }
    //for (int j = 0; j < N; ++j)
      //std::cout << a[j] << (j == N - 1 ? "\n" : " ");
  }
}
