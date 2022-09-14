// Test simple indirect memory access

#include "../Common/interface.h"
#include "../Common/test.h"

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
} args_;

void ind2dread(int64_t *__restrict a, int64_t *__restrict b, int64_t *__restrict c,
               int64_t *__restrict begin) {
  #pragma ss config
  {
    #pragma ss stream
    for (int64_t i = 0; i < N; ++i) {
      #pragma ss dfg unroll(1)
      for (int64_t j = 0; j < N; ++j) {
        c[i * N + j] = a[j + begin[i]] + b[i * N + j];
      }
    }
  }
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *begin = args->begin;
  int64_t *c = args->ref;
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      c[i * N + j] = a[j + begin[i]] + b[i * N + j];
    }
  }
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  ind2dread(args->a, args->b, args->c, args->begin);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N * N, "%ld");
  return 1;
}

struct Arguments *init_data() {
  init_linear(args_.a, N * 2);
  init_linear(args_.b, N * N);
  init_rand(args_.begin, N);
  return &args_;
}

