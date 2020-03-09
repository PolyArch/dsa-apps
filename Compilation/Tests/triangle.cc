// Test of the triangular memory stream

#include <stdio.h>

#include "../Common/test.h"

#define N 8

double a[N * N];
double b[N * N];
double c[N * N];
double ref[N * N];

int main() {
  init<double>(a, N * N);
  init<double>(b, N * N);
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j)
      ref[i * N + j] = a[i * N + j] + b[i * N + j];

  begin_roi();
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(4)
      for (int j = i; j < N; ++j) {
        c[i * N + j] = a[i * N + j] + b[i * N + j];
      }
    }
  }
  end_roi();
  sb_stats();

  comp<double>(ref, c, N * N);
  return 0;
}
