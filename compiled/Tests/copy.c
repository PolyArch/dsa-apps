// Test on data copy, the memcpy intrin.

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
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
  init_odd(args_.a, N);
  init_even(args_.b, N);
  return &args_;
}

void copy(double *a, double *b) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg
    for (int i = 0; i < N; ++i) {
      b[i] = a[i];
    }

  }
}

void run_reference(struct Arguments *args) {
  for(int i = 0; i < N; ++i){
    args->ref[i] = args->a[i];
  }
}

void run_accelerator(struct Arguments *args, int x) {
  copy(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->b, args->ref, N, "%lf");
  return 1;
}
