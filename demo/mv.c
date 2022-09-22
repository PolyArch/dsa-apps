// Test of writing a vector of accumulators to the memory
#include <stdio.h>
#include "common/test.h"
#include "common/timing.h"
#include "common/spatial_intrin.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif


struct Arguments {
  int64_t a[N * N], b[N], c[N], d[N], ref[N];
} args_;

void mv(int64_t *a, int64_t *b, int64_t *c) {
#pragma ss config
  {
    arrayhint(a, /*array-size*/N * N * sizeof(int64_t), /*analyzed-reuse*/-1);
    arrayhint(b, /*array-size*/N * sizeof(int64_t),     /*analyzed-reuse*/-1);
    arrayhint(c, /*array-size*/N * sizeof(int64_t),     /*analyzed-reuse*/-1);
#pragma ss stream
    for (int i = 0; i < N; ++i) {
      int64_t v = 0;
#pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j) {
	v += a[i * N + j] * b[j];
      }
      c[i] = v;
    }
  }
}

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  {
    int64_t *a = args->a;
    int64_t *b = args->b;
    int64_t *c = args->ref;
    for (int i = 0; i < N; ++i) {
      int64_t v = 0;
      for (int j = 0; j < N; ++j) {
	v += a[i * N + j] * b[j];
      }
      c[i] = v;
    }
  }
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  mv(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

