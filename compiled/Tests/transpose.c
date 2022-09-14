// The first poc

#include <stdio.h>
#include <string.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  double a[N * N], b[N * N], ref[N * N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double *ref = args->ref;
  unsigned long start = rdcycle();
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i * N + j] = a[j * N + i];
}

void transpose(double *a, double *b) {
  #pragma ss config
  {
    for (int i = 0; i < N; ++i) {
      #pragma ss stream
      #pragma ss dfg
      for (int j = 0; j < N; ++j)
        b[i * N + j] = a[j * N + i];
    }
  }
}

void run_accelerator(struct Arguments *args) {
  transpose(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->ref, args->b, N, "%lf");
  return 1;
}

