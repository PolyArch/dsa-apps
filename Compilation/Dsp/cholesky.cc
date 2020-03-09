#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include <sim_timing.h>

#ifndef N
#define N 32
#endif

#ifndef U
#define U 2
#endif

void cholesky(double *a, double *L) {
  #pragma ss config
  {
    for (int i = 0; i < N - 1; ++i) {

      double sqrt_inv, inv, aii = a[i * (N + 1)];
      #pragma ss dfg temporal
      {
        sqrt_inv = 1.0 / sqrt(aii);
        inv = 1.0 / aii;
      }

      #pragma ss stream nonblock
      #pragma ss dfg dedicated
      for (int j = i; j < N; ++j)
        L[j * N + i] = a[i * N + j + 1] * sqrt_inv;

      #pragma ss stream
      for (int j = i + 1; j < N; ++j) {
        #pragma ss dfg dedicated unroll(U)
        for (int k = j; k < N; ++k) {
          a[j * N + k] -= a[i * N + j] * a[i * N + k + 1] * inv;
        }
      }

    }
  }
}


double a[N * N], L[N * N];

int main() {
  for (int i = 0; i < N * N; ++i) {
    a[i] = rand();
  }
  cholesky(a, L);
  begin_roi();
  cholesky(a, L);
  end_roi();
  sb_stats();
  return 0;
}
