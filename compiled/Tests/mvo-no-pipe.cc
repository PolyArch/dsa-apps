// This is a trivial solution to mvo
// It introduces the problem of no preheader in the second sub-dfg

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "../Common/test.h"

#ifndef N
#define N 32
#endif

double a[N * N], b[N], ref[N * N], v[N];
double aa[N * N], bb[N], vv[N];

void kernel(double *a, double *b, double *v) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      double acc = 0.0;
      #pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j) {
        acc += a[i * N + j] * b[j];
      }
      v[i] = acc;
    }

    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j) {
        a[i * N + j] -= v[i] * b[j];
      }
    }
  }
}

int main() {

  init<double>(a, N * N);
  init<double>(b, N);

  memcpy(ref, a, sizeof a);
  for (int i = 0; i < N; ++i) {
    double v = 0.0;
    for (int j = 0; j < N; ++j)
      v += ref[i * N + j] * b[j];
    for (int j = 0; j < N; ++j)
      ref[i * N + j] -= v * b[j];
  }

  kernel(aa, bb, vv);
  begin_roi();
  kernel(a, b, v);
  end_roi();
  sb_stats();

  comp<double>(a, ref, N * N);
  return 0;
}
