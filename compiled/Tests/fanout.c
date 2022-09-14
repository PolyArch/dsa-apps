// Manual unrolling with pragma unrolling.

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N], b[N], ref[N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->ref;
  int64_t *b = args->b;
  for (int i = 0; i < N - 2; ++i) {
    a[i] = b[i] + b[i + 1] + b[i + 2];
  }
}

void fanout(int64_t *__restrict a, int64_t *__restrict b) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated unroll(1)
    for (int i = 0; i < N - 2; ++i) {
      a[i] = b[i] + b[i + 1] + b[i + 2];
    }
  }
}

void run_accelerator(struct Arguments *args) {
  fanout(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->a, args->ref, N - 2, "%ld");
  return 1;
}

