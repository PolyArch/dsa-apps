// Test of stretched repeat

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N * N], b[N * N], c[N * N], ref[N * N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.b, N * N);
  return &args_;
}


void sr(int64_t *a, int64_t *b, int64_t *c) {
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
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j) {
      ref[i * N + j] = a[i * N + j] + b[i];
    }
}

void run_accelerator(struct Arguments *args) {
  sr(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->ref, args->c, N * N, "%ld");
  return 1;
}
