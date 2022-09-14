// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 1024
#endif

struct Arguments {
  double a[N], b[N], ref[N];
} args_;

struct Arguments *init_data() {
  // data initialization
  init_linear(args_.a, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double *ref = args->ref;
  for(int i = 0; i < N; ++i) {
    ref[i] = a[i];
  }
}

void move(double *a, double *b) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int i = 0; i < N; ++i) {
      b[i] = a[i];
    }
  }
}

void run_accelerator(struct Arguments *args) {
  move(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->a, args->b, N, "%f");
  return 1;
}

