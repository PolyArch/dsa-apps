// Test of vector update

#include <stdio.h>
#include <string.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 16
#else
#define N 16
#endif

struct Arguments {
  int64_t a[N * N], b[N], c[N], d[N], ref[N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      ref[j] += a[i * N + j] * b[j];
    }
  }
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}


void update_small(int64_t *a, int64_t *b, int64_t *c) {
  #pragma ss config
  {
    int nb = 8;
    for (int jo = 0; jo < N; jo += nb) {
      #pragma ss stream
      for (int i = 0; i < N; ++i) {
        #pragma ss dfg dedicated unroll(4)
        for (int ji = 0; ji < nb; ++ji) {
          int j = jo + ji;
          c[j] += a[i * N + j] * b[j];
        }
      }
    }
  }
}

void run_accelerator(struct Arguments *args, int iswarmup) {
  update_small(args->a, args->b, args->c);
  if (iswarmup) {
    memset(args->a, 0, sizeof args->a);
  }
}

