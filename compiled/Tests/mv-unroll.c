// Test of writing a vector of accumulators to the memory

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N * N], b[N], c[N], d[N], ref[N];
} args_;

struct Arguments *init_data() {
  // data initialization
  init_linear(args_.a, N * N);
  init_linear(args_.b, N);
  return &args_;
}

void mv_unroll(int64_t *a, int64_t *b, int64_t *c) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; i += 4) {
      int64_t v0 = 0;
      int64_t v1 = 0;
      int64_t v2 = 0;
      int64_t v3 = 0;
      #pragma ss dfg dedicated unroll(1)
      for (int j = 0; j < N; ++j) {
        v0 += a[i / 4 * N + j * 4 + 0] * b[j];
        v1 += a[i / 4 * N + j * 4 + 1] * b[j];
        v2 += a[i / 4 * N + j * 4 + 2] * b[j];
        v3 += a[i / 4 * N + j * 4 + 3] * b[j];
      }
      c[i + 0] = v0;
      c[i + 1] = v1;
      c[i + 2] = v2;
      c[i + 3] = v3;
    }
  }
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *c = args->c;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; i += 4) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < 4; ++k) {
        ref[i + k] += a[i / 4 * N + j * 4 + k] * b[j];
      }
    }
  }
}

void run_accelerator(struct Arguments *args) {
  mv_unroll(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}


