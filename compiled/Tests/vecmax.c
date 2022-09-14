// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 1024
#endif

struct Arguments {
  int64_t a[N], b[N], c[N], ref[N];
} args_;

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    ref[i] = max64(a[i], b[i]);
  }
}

void vecmax(int64_t *a, int64_t *b, int64_t *c) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int i = 0; i < N; ++i) {
      c[i] = max64(a[i], b[i]);
    }
  }
}

void run_accelerator(struct Arguments *args) {
  vecmax(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

struct Arguments *init_data() {
  // data initialization
  init_rand(args_.a, N);
  init_rand(args_.b, N);
  return &args_;
}

