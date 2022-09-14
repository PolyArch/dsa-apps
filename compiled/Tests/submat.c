// Test of strided access pattern

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

void submat(int64_t *a, int64_t *b, int64_t *c) {
#pragma ss config
  {
#pragma ss stream
    for (int i = 1; i < N - 1; ++i) {
#pragma ss dfg dedicated unroll(2)
      for (int j = 1; j < N - 1; ++j) {
	c[i * N + j] = a[i * N + j] + b[i * N + j];
      }
    }
  }
}

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.b, N * N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 1; i < N - 1; ++i)
    for (int j = 1; j < N - 1; ++j)
      ref[i * N + j] = a[i * N + j] + b[i * N + j];
}

void run_accelerator(struct Arguments *args) {
  submat(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->ref, args->c, N * N, "%ld");
  return 1;
}
