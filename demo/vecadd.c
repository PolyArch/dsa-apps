// The first proof of concept

#include <stdio.h>

#include "common/test.h"
#include "common/timing.h"

#define N 1024

#ifndef TYPE
#define TYPE int64_t
#endif

struct Arguments {
  TYPE a[N], b[N], c[N], ref[N];
} args_;

// Invoke host reference
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
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

// Invoke accelerator
void run_accelerator(struct Arguments *args) {
  vecadd(args->a, args->b, args->c);
}

// compare the results
int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

struct Arguments *init_data() {
  // data initialization
  init_odd(args_.a, N);
  init_even(args_.b, N);
  return &args_;
}

