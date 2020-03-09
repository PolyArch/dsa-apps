// Test of stretched repeat

#include <stdio.h>

#include "../Common/test.h"

#define N 16

int64_t a[N * N];
int64_t b[N * N];
int64_t c[N * N];
int64_t ref[N * N];

int main() {
  init(a, N * N);
  init(b, N * N);
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j) {
      ref[i * N + j] = a[i * N + j] + b[i];
    }

  begin_roi();
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(4)
      for (int j = i; j < N; ++j) {
        c[i * N + j] = a[i * N + j] + b[i];
      }
    }
  }
  end_roi();
  sb_stats();

  comp(ref, c, N * N);
  return 0;
}
