// Manual loop unrolling without pragma unrolling.

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 1024
#endif

struct Arguments {
  int64_t a[N], b[N], c[N], ref[N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->b;
  int64_t *ref = args->ref;
  for (int i = 0; i < N; i += 2) {
    ref[i + 0] = a[i + 0] + b[i + 0];
    ref[i + 1] = a[i + 1] + b[i + 1];
  }
}

void slp(int64_t *a, int64_t *b, int64_t *c) {
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

void run_accelerator(struct Arguments *args) {
  slp(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

