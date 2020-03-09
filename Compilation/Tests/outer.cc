// Test of repeat port

#include <stdio.h>

#include "../Common/test.h"

#define N 16

double a[N], b[N], c[N * N], ref[N * N];

void kernel() {
  #pragma ss config
  {

    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j)
        c[i * N + j] = a[i] * b[j];
    }
  }
}

int main() {

  init<double>(a, N);
  init<double>(b, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i * N + j] = a[i] * b[j];

  kernel();
  begin_roi();
  kernel();
  end_roi();
  sb_stats();

  comp<double>(c, ref, N * N);
  return 0;
}
