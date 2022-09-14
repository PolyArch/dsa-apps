// Test simple indirect memory access

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
  #define n 32
  #define N 1024
#else
  #define n 4
  #define N 16
#endif

struct Arguments {
  int64_t a[N], b[N], c[N], d[N], ref[N];
} args_;

struct Arguments *init_data() {
  // data initialization
  init_rand(args_.a, N);
  init_rand(args_.b, N);
  init_linear(args_.c, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *c = args->c;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    ref[i] = c[a[i]] + c[b[i]];
  }
}

void ind1d2d(int64_t *a, int64_t *b, int64_t *c, int64_t *d) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < n; ++i) {
      #pragma ss dfg unroll(1)
      for (int j = 0; j < n; ++j) {
        d[i * n + j] = c[a[i * n + j]] + c[b[i * n + j]];
      }
    }
  }
}

void run_accelerator(struct Arguments *args) {
  ind1d2d(args->a, args->b, args->c, args->d);
}

int sanity_check(struct Arguments *args) {
  compare(args->d, args->ref, N, "%ld");
  return 1;
}

