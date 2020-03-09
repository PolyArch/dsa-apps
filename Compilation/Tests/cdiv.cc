// Test of ceiling division to compute the repeat#

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "../Common/test.h"

#ifndef N
#define N 3
#endif

double coef;
double a[N * N], b[N], c[N], cc[N], ref[N];

void kernel(double *a, double *b, double *c, double coef) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      double v = 0.0;
      #pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j)
        v += a[i * N + j] * b[j] * coef;
      c[i] = v;
    }
  }
}

int main() {

  coef = 3;

  init<double>(a, N * N);
  init<double>(b, N);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i] += a[i * N + j] * b[j] * coef;

  kernel(a, b, cc, coef);
  begin_roi();
  kernel(a, b, c, coef);
  end_roi();
  sb_stats();

  comp<double>(c, ref, N);
  return 0;
}
