// Test of communication between temporal and dedicated region

#include <stdio.h>
#include <math.h>

#include "../Common/spatial_inrin.h"
#include "../Common/test.h"
#include "../Common/timing.h"

#define N 16

struct Arguments {
  double a[N];
  double b[N];
} args_;

void norm(double *b) {
  #pragma ss config
  {
    double acc = 0.0;
    #pragma ss stream nonblock
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i)
      acc += b[i] * b[i];

    #pragma ss dfg temporal
    {
      acc = fsqrt(1.0 / acc);
    }

    #pragma ss stream
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i)
      b[i] *= acc;
  }
}

struct Arguments *init_data() {
  init_linear(args_.a, N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double len = 0.0;
  for (int i = 0; i < N; ++i)
    len += a[i] * a[i];
  len = sqrt(1.0 / len);
  for (int i = 0; i < N; ++i)
    a[i] *= len;
}

void run_accelerator(struct Arguments *args, int iswarmup) {
  norm(args->b);
  if (iswarmup) {
    init_linear(args_.b, N);
  }
}

int sanity_check(struct Arguments *args) {
  compare(args->a, args->b, N, "%f");
  return 1;
}

