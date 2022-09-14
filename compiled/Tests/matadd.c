// Test of 2-d array access

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  double a[N * N], b[N * N], c[N * N], ref[N * N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.b, N * N);
  return &args_;
}

void matadd(double *a, double *b, double *c){
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      #pragma ss dfg dedicated unroll(2)
      for (int j = 0; j < N; ++j) {
        c[i * N + j] = a[i * N + j] + b[i * N + j];
      }
    }
  }
} 

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double *b = args->b;
  double *ref = args->ref;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i * N + j] = a[i * N + j] + b[i * N + j];
}

void run_accelerator(struct Arguments *args) {
  matadd(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->ref, args->c, N * N, "%lf");
  return 1;
}
