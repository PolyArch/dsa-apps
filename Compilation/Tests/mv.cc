// Test of writing a vector of accumulators to the memory

#include <algorithm>
#include <stdio.h>

#include "../Common/test.h"

#define N 128

double a[N * N], b[N], c[N], d[N], ref[N];

void kernel(double *a, double *b, double *c) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      double v = 0.0;
      #pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j) {
        v += a[i * N + j] * b[j];
      }
      c[i] = v;
    }
  }
}

int main() {

  init<double>(a, N);
  init<double>(b, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i] += a[i * N + j] * b[j];

  kernel(a, b, d);
  begin_roi();
  kernel(a, b, c);
  end_roi();
  sb_stats();

  comp<double>(c, ref, N);
  return 0;
}
