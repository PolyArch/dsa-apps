// Manual unrolling with pragma unrolling.

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 1024
#endif

struct Argument {
  double a[N], b[N], c[N], ref[N];
} args_;

struct Argument *init_data() {
  // data initialization
  init_linear(args_.a, N);
  init_linear(args_.b, N);
  return &args_;
}

void coal(double *a, double *b, double *c) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int i = 0; i < N; i += 2) {
      c[i + 0] = a[i + 0] + b[i + 0];
      c[i + 1] = a[i + 1] + b[i + 1];
    }
  }
}

void run_reference(struct Argument *args) {
  double *a = args->a;
  double *b = args->b;
  double *c = args->ref;
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
}

void run_accelerator(struct Argument *args) {
  coal(args->a, args->b, args->c);
}

void sanity_check(struct Argument *args) {
  compare(args->c, args->ref, N, "%lf");
}

