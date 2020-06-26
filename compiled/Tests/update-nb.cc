// Test of non-blocking stream update

#include <algorithm>
#include <stdio.h>

#include "../Common/test.h"

#ifndef N
#define N 128
#endif

double a[N * N], b[N], c[N], d[N], ref[N];
double aa[N * N], bb[N];

void kernel(double *a, double *b, double *c) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j) {
        c[j] += a[i * N + j] * b[j];
      }
    }
  }
}

int main() {

  init<double>(a, N);
  init<double>(b, N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      ref[j] += a[i * N + j] * b[j];
    }
  }

  kernel(aa, bb, d);
  begin_roi();
  kernel(a, b, c);
  end_roi();
  sb_stats();

  comp<double>(c, ref, N);
  return 0;
}
