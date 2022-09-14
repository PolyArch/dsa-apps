// Test of stretched sum

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N * N], b[N * N], ref[N * N];
  int64_t coef;
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  args_.coef = 114514;
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *ref = args->ref;
  int64_t coef = args->coef;
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j)
      ref[i * N + j] = a[i * N + j] * coef;
}

void sr_sum(int64_t *a, int64_t *b, int64_t coef){
#pragma ss config
  {
#pragma ss stream
    for (int i = 0; i < N; ++i) {
#pragma ss dfg dedicated unroll(4)
      for (int j = i; j < N; ++j) {
	b[i * N + j] = a[i * N + j] * coef;
      }
    }
  }
}

void run_accelerator(struct Arguments *args) {
  sr_sum(args->a, args->b, args->coef);
}

int sanity_check(struct Arguments *args) {
  compare(args->b, args->ref, N * N, "%ld");
  return 1;
}

