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

#ifndef TYPE
#define TYPE int64_t
#endif

#if TYPE == int64_t
#define TY_FMT "%ld"
#elif TYPE == int32_t
#define TY_FMT "%d"
#elif TYPE == double || TYPE == float
#define TY_FMT "%f"
#endif

struct Arguments {
  TYPE a[N], b[N], c[N], ref[N];
} args_;

void run_reference(struct Arguments *args) {
  TYPE *a = args->a;
  TYPE *b = args->b;
  TYPE *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    ref[i] = a[i] + b[i];
  }
}

void vecadd(TYPE *a, TYPE *b, TYPE *c) {
  #pragma ss config
  {
    // arrayhint(a, N * sizeof(TYPE), 0);
    // arrayhint(b, N * sizeof(TYPE), 0);
    // arrayhint(c, N * sizeof(TYPE), 0);
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

void run_accelerator(struct Arguments *args) {
  vecadd(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, TY_FMT);
  return 1;
}

struct Arguments *init_data() {
  // data initialization
  init_odd(args_.a, N);
  init_even(args_.b, N);
  return &args_;
}

