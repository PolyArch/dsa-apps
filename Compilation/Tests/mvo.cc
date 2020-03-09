// Test of the idiom, produce-and-reuse

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "../Common/test.h"

#ifndef N
#define N 32
#endif

double a[N * N], b[N], ref[N * N];
double aa[N * N], bb[N];

void kernel(double *a, double *b) {
  #pragma ss config
  {
    double v = 0;
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j) {
        v += a[i * N + j] * b[j];
      }
      #pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j) {
        a[i * N + j] -= v * b[j];
      }
    }
  }
}

int main() {

  init<double>(a, N * N);
  init<double>(b, N);

  memcpy(ref, a, sizeof a);
  for (int i = 0; i < N; ++i) {
    double v = 0;
    for (int j = 0; j < N; ++j)
      v += ref[i * N + j] * b[j];
    for (int j = 0; j < N; ++j)
      ref[i * N + j] -= v * b[j];
  }

  kernel(aa, bb);
  begin_roi();
  kernel(a, b);
  end_roi();
  sb_stats();

  comp<double>(a, ref, N * N);
  return 0;
}
