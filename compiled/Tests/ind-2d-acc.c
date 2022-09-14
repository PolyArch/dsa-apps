// Test simple indirect memory access

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 64
#endif


struct Arguments {
  int64_t a[N * 2];
  int64_t b[N * N];
  int64_t c[N * N];
  int64_t ref[N * N];
  int64_t begin[N];
  int64_t n[N];
} args_;

void ind_2d_acc(int64_t *__restrict a, int64_t *__restrict b,
    int64_t *__restrict begin, int64_t *__restrict n) {
#pragma ss config
  {
#pragma ss stream
    for (int64_t i = 0; i < N; ++i) {
      int64_t acc = 0;
#pragma ss dfg unroll(1)
      for (int64_t j = 0; j < n[i]; ++j) {
	acc += a[begin[i] + j];
      }
      b[i] = acc;
    }
  }
}

struct Arguments* init_data() {
  int64_t *a = args_.a;
  int64_t *b = args_.b;
  int64_t *begin = args_.begin;
  int64_t *n = args_.n;
  init_linear(a, N * 2);
  init_linear(b, N * N);
  for (int i = 0; i < N; ++i) {
    begin[i] = rand() % N;
    n[i] = rand() % N + 2;
  }
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *begin = args->begin;
  int64_t *n = args->n;
  int64_t *ref = args->ref;
  for (int64_t i = 0; i < N; ++i) {
    int64_t acc = 0;
    for (int64_t j = 0; j < n[i]; ++j) {
      acc += a[j + begin[i]];
    }
    ref[i] = acc;
  }
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  ind_2d_acc(args->a, args->b, args->begin, args->n);
}

void sanity_check(struct Arguments *args) {
  compare(args->b, args->ref, N, "%ld");
}

