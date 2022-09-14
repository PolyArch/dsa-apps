// Test of repeat port

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N], b[N], c[N * N], ref[N * N];
} args_;

void outer(int64_t *a, int64_t *b, int64_t *c) {
#pragma ss config
  {
#pragma ss stream
    for (int i = 0; i < N; ++i) {
#pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j)
	c[i * N + j] = a[i] * b[j];
    }
  }
}


struct Arguments *init_data() {
  init_linear(args_.a, N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i * N + j] = a[i] * b[j];
}

void run_accelerator(struct Arguments *args) {
  outer(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N * N, "%ld");
  return 1;
}

