// FIXME(@were): Fix the correctness?
// Test of vector update

#include <stdio.h>
#include <string.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 32
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N * N], b[N], c[N], ref[N];
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

void update(int64_t *a, int64_t *b, int64_t *c) {
  #pragma ss config
  {
    int nb = 16;
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
  update(args->a, args->b, args->c);
  if (iswarmup) {
    memset(args->a, 0, sizeof(args->a));
  }
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

