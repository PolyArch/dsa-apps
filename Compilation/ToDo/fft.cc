#include <complex>
#include <algorithm>
#include <stdio.h>

#include "../Common/test.h"

using std::complex;
using std::swap;

#define N 128

complex<float> *fft(complex<float> *a, complex<float> *b, complex<float> *w) {

  #pragma ss config
  {
    for (int blocks = N / 2; blocks; blocks >>= 1) {
      int span = N / blocks;
      #pragma ss stream
      for (int j = 0; j < span / 2 * blocks; j += blocks) {
        #pragma ss dfg dedicated
        for (int i = 0; i < blocks; ++i) {
          //printf("%d %d %d\n", blocks, j, i);
          complex<float> &L = a[2 * j + i];
          complex<float> &R = a[2 * j + i + blocks];
          complex<float> tmp(w[j] * L);
          b[i + j] = L + tmp;
          b[i + j + span / 2 * blocks] = R - tmp;
        }
      }
      swap(a, b);
    }
  }
  return a;
}

complex<float> a[N], b[N], w[N];

int main() {
  fft(a, b, w);
}
