#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"
#include "../Specs/cholesky.h"

struct Arguments {
  TYPE a[N * N], L[N * N];
  TYPE a_[N * N], L_[N * N];
} args_;

struct Arguments *init_data() {
  for (int i = 0; i < N * N; ++i) {
    args_.a[i] = rand() % 32768;
  }
  return &args_;
}

void cholesky(TYPE *a, TYPE *L) {
  #pragma ss config
  {
    arrayhint(a, N * N * sizeof(TYPE), 1.0 - (double) (N * N) / ((N * (N + 1) * (2 * N + 1)) / 6.0));
    arrayhint(L, N * N * sizeof(TYPE), 0);
    for (int i = 0; i < N - 2; ++i) {

      TYPE sqrt_inv, inv, aii = a[i * (N + 1)];
      #pragma ss dfg temporal
      {
        sqrt_inv = 1.0 / fsqrt(aii);
        inv = 1.0 / aii;
      }

      #pragma ss stream nonblock
      #pragma ss dfg dedicated
      for (int j = i; j < N; ++j)
        L[j * N + i] = a[i * N + j + 1] * sqrt_inv;

      #pragma ss stream
      for (int j = i + 1; j < N; ++j) {
        #pragma ss dfg dedicated unroll(2)
        for (int k = j; k < N; ++k) {
          a[j * N + k] -= a[i * N + j] * a[i * N + k + 1] * inv;
        }
      }

    }
  }
}

void run_accelerator(struct Arguments *args, int iswarmup) {
  if (iswarmup) {
    cholesky(args->a, args->L);
  } else {
    cholesky(args->a_, args->L_);
  }
}

NO_SANITY_CHECK
