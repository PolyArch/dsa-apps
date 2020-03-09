// Test of vector update

#include <algorithm>
#include <stdio.h>

#include "../Common/test.h"

#ifndef N
#define N 128
#endif

double a[N * N], b[N], c[N], d[N], ref[N];

void kernel(double *a, double *b, double *c) {
  #pragma ss config
  {
    int nb = 8;
    for (int jo = 0; jo < N; jo += nb) {
      #pragma ss stream
      for (int i = 0; i < N; ++i) {
        #pragma ss dfg dedicated unroll(4)
        for (int ji = 0; ji < nb; ++ji) {
          int j = jo + ji;
          c[j] += a[i * N + j] * b[j];
        }
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

  kernel(a, b, d);
  begin_roi();
  kernel(a, b, c);
  end_roi();
  sb_stats();

  comp<double>(c, ref, N);
  return 0;
}
