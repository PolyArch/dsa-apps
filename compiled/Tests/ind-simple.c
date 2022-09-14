// Test simple indirect memory access

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N], b[N], c[N], d[N], ref[N];
} args_;

struct Arguments *init_data() {
  init_rand(args_.a, N);
  init_rand(args_.b, N);
  init_linear(args_.c, N);
  return &args_;
}

void ind_simple(int64_t * __restrict__ a, int64_t * __restrict__ b,
                int64_t * __restrict__ c, int64_t * __restrict__ d) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg unroll(1)
    for (int64_t i = 0; i < N; ++i) {
      d[i] = c[a[i]] + c[b[i]];
    }
  }
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

void run_accelerator(struct Arguments *args) {
  ind_simple(args->a, args->b, args->c, args->d);
}

int sanity_check(struct Arguments *args) {
  compare(args->d, args->ref, N, "%ld");
  return 1;
}

