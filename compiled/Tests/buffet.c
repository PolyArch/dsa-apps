// Test buffet reuse.

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#define N 32

struct Arguments {
  int64_t a[N * N], b[N], c[N], ref[N];
} args_;

struct Arguments *init_data() {
  // data initialization
  init_linear(args_.a, N * N);
  init_linear(args_.b, N);
  return &args_;
}

// FIXME(@were): This is not working because of reset/produce level analysis.
void buffet(int64_t *a, int64_t *b, int64_t *c) {
  #pragma ss config
  {
    #pragma ss stream
    for (int64_t i = 0; i < N; ++i) {
      int64_t acc = 0;
      for (int64_t j = 0; j < N; ++j) {
        #pragma ss dfg dedicated unroll(4)
        for (int64_t k = 0; k < N; ++k) {
          acc += a[i * N + k] * b[k];
        }
      }
      c[i] = acc;
    }
  }
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; ++i) {
    int64_t acc = 0;
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        acc += a[i * N + k] * b[k];
      }
    }
    ref[i] = acc;
  }
}

void run_accelerator(struct Arguments *args) {
  buffet(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}
