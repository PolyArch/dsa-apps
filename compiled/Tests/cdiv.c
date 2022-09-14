// Test of ceiling division to compute the repeat#

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 127
#else
#define N 3
#endif

struct Argument {
  int64_t coef;
  int64_t a[N * N], b[N], c[N], ref[N];
} args_;

struct Argument *init_data() {
  args_.coef = 3;
  init_linear(args_.a, N * N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Argument *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  int64_t coef = args->coef;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i] += a[i * N + j] * b[j] * coef;
}

void cdiv(int64_t *a, int64_t *b, int64_t *c, int64_t coef) {
#pragma ss config
  {
#pragma ss stream
    for (int i = 0; i < N; ++i) {
      int64_t v = 0;
#pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j)
	v += a[i * N + j] * b[j];
      c[i] = v * coef;
    }
  }
}

void run_accelerator(struct Argument *args, int is_warmup) {
  cdiv(args->a, args->b, args->c, args->coef);
}

int sanity_check(struct Argument *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

