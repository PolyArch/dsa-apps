// TODO(@were): Fix the correctness.

// Test of communication between temporal and dedicated region

#include <stdio.h>
#include <math.h>

#include "../Common/spatial_inrin.h"
#include "../Common/test.h"
#include "../Common/timing.h"

#define N 16

struct Arguments {
  double a[N * N];
  double b[N * N];
} args_;

void norm(double *a) {
  #pragma ss config
  {
    double acc = 0.0;
    #pragma ss stream nonblock
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated
      for (int j = 0; j < N; ++j) {
        acc += a[i * N + j] * a[i * N + j];
      }
    }

    #pragma ss dfg temporal
    {
      acc = fsqrt(1.0 / acc);
    }

    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated
      for (int j = 0; j < N; ++j) {
        a[i * N + j] *= acc;
      }
    }
  }
}


struct Arguments *init_data() {
  return &args_;
}

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double len = 0.0;
  for (int i = 0; i < N * N; ++i)
    len += a[i] * a[i];
  len = sqrt(1.0 / len);
  for (int i = 0; i < N * N; ++i)
    a[i] *= len;
}

void run_accelerator(struct Arguments *args) {
  norm(args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->a, args->b, N, "%f");
  return 1;
}

