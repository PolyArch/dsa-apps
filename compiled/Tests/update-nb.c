// Test of non-blocking stream update

#include <string.h>
#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 32
#else
#define N 32
#endif

struct Arguments {
  double a[N * N], b[N], c[N], d[N], ref[N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double *b = args->b;
  double *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      ref[j] += a[i * N + j] * b[j];
    }
  }
}

void update_nb(double *a, double *b, double *c) {
#pragma ss config
  {
#pragma ss stream nonblock
    for (int i = 0; i < N; ++i) {
#pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j) {
	c[j] += a[i * N + j] * b[j];
      }
    }
  }
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  update_nb(args->a, args->b, args->c);
  if (is_warmup) {
    memset(args->a, 0, sizeof args->a);
  }
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%lf");
  return 1;
}

