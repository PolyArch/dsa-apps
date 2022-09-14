// Test of scratchpad

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"

#ifdef LONG_TEST
#define N 512
#else
#define N 128
#endif

struct Arguments {
  int64_t a[N], b[N], c[N], ref[N];
} args_;

struct Arguments *init_data() {
  // data initialization
  init_linear(args_.a, N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    ref[i] = a[i] + b[i];
  }
}

void spad(int64_t *a, int64_t *b, int64_t *c) {
  #pragma ss config
  {
    int64_t l1_a[N];
    int64_t l1_b[N];

    // spadoverride(l1_b, 1048576, sizeof(l1_b));

    #pragma ss stream
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i) {
      l1_a[i] = a[i];
    }

    #pragma ss stream
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i) {
      l1_b[i] = b[i];
    }

    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int i = 0; i < N; ++i) {
      c[i] = l1_a[i] + l1_b[i];
    }

  }
}

void run_accelerator(struct Arguments *args) {
  spad(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

