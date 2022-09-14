// FIXME(@were): The correctness is not done.

// Test of the idiom, produce-and-reuse

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N * N], b[N], ref[N * N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.ref, N * N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args){
  int64_t *ref = args->ref;
  int64_t *b = args->b;
  for (int i = 0; i < N; ++i) {
    int64_t v = 0;
    for (int j = 0; j < N; ++j)
      v += ref[i * N + j] * b[j];
    for (int j = 0; j < N; ++j)
      ref[i * N + j] -= v * b[j];
  }
}

void mvo(int64_t *a, int64_t *b) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      int64_t v = 0;
      #pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j) {
        v += a[i * N + j] * b[j];
      }
      #pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j) {
        a[i * N + j] -= v * b[j];
      }
    }
  }
}

void run_accelerator(struct Arguments *args, int iswarmup) {
  mvo(args->a, args->b);
  if (iswarmup) {
    init_linear(args->a, N * N);
  }
}

int sanity_check(struct Arguments *args) {
  compare(args->a, args->ref, N * N, "%ld");
  return 1;
}

