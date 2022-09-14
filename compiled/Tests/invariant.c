// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 36
#endif


struct Arguments {
  int64_t a[N], b[2], c[N], ref[N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N);
  init_linear(args_.b, 2);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    ref[i] = a[i] * b[0] * b[1];
  }
}

void invariant(int64_t *a, int64_t *b, int64_t *c) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] * b[0] * b[1];
    }
  }
}

void run_accelerator(struct Arguments *args) {
  invariant(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

