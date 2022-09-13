#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "../Common/spatial_inrin.h"
#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Specs/solver.h"


#ifndef N
#define N 48
#endif

void solver(TYPE *a, TYPE *v) {
  #pragma ss config
  {
    arrayhint(a, N * N * sizeof(TYPE), 0);
    arrayhint(v, N * sizeof(TYPE), 1.0 - (double) N / ((1 + N) * N / 2.0));
    for (int i = 0; i < N - 1; ++i) {
      TYPE vv = 0;
      TYPE v0 = v[i];
      TYPE a0 = a[i * N + i];
      #pragma ss dfg temporal
      {
        vv = v0 / a0;
      }
      // v[i] = vv;
      #pragma ss stream
      #pragma ss dfg dedicated unroll(4)
      for (int j = i + 1; j < N; ++j) {
        v[j] -= a[i * N + j] * vv;
      }
    }
  }
}

struct Arguments {
  TYPE a[N * N], v[N];
  TYPE a_[N * N], v_[N];
} args_;

struct Arguments *init_data() {
  return &args_;
}

void run_reference(struct Arguments *args) {
}

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    solver(args->a, args->v);
  } else {
    solver(args->a_, args->v_);
  }
}

int sanity_check(struct Arguments *_) {
  return 1;
}

