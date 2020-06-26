// Test of stretched sum

#include <stdio.h>

#include "../Common/test.h"

#define N 8

double a[N * N];
double b[N * N];
double ref[N * N];

int main() {
  double coef = 114.514 / 1919.810;

  init<double>(a, N * N);

  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j)
      ref[i * N + j] = a[i * N + j] * coef;

  begin_roi();
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(2)
      for (int j = i; j < N; ++j) {
        b[i * N + j] = a[i * N + j] * coef;
      }
    }
  }
  end_roi();
  sb_stats();

  comp<double>(ref, b, N * N);
  return 0;
}
