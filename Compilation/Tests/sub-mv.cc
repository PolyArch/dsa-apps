// Test of imperfect loop repeat
// Test of non-all-related loop nests

#include <algorithm>
#include <stdio.h>

#include "../Common/test.h"

#define N 16

double a[N * N], b[N], c[N], d[N], ref[N];

void kernel(double *a, double *b, double *c) {
  #pragma ss config
  {
    for (int k = 0; k < N; ++k) {
      #pragma ss stream
      for (int i = k; i < N; ++i) {
        double v = 0.0;
        #pragma ss dfg dedicated unroll(4)
        for (int j = k; j < N; ++j) {
          v += a[i * N + j] * b[j];
        }
        c[i] = v;
      }
    }
  }
}

int main() {

  init<double>(a, N);
  init<double>(b, N);

  for (int k = 0; k < N; ++k) {
    for (int i = k; i < N; ++i) {
      ref[i] = 0;
      for (int j = k; j < N; ++j) {
        ref[i] += a[i * N + j] * b[j];
      }
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
