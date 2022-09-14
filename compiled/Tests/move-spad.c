// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 512
#else
#define N 128
#endif

struct Arguments {
  double a[N], b[N];
  double ref[N];
} args_;

void move_spad(double *a, double *b) {
  #pragma ss config
  {
    double b_l1[N];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int i = 0; i < N; ++i) {
      b_l1[i] = a[i];
    }
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int i = 0; i < N; ++i) {
      b[i] = b_l1[i];
    }
  }
}

struct Arguments *init_data() {
  init_linear(args_.a, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double *b = args->ref;
  for(int i = 0; i < N; ++i) b[i] = a[i];
}

void run_accelerator(struct Arguments *args) {
  move_spad(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->a, args->b, N, "%f");
  return 1;
}
