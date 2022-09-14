// Memory coalescing with code strided overlapping.

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
  int64_t a[N], b[N], ref[N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->ref;
  for (int64_t i = 0; i + 3 < N; ++i) {
    b[i] = a[i] + a[i + 1] + a[i + 2] + a[i + 3];
  }
}

void fansum(int64_t *__restrict a, int64_t *__restrict b) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated
    for (int64_t i = 0; i + 3 < N; ++i) {
      b[i] = a[i] + a[i + 1] + a[i + 2] + a[i + 3];
    }
  }
}

void run_accelerator(struct Arguments *args) {
  fansum(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->b, args->ref, N, "%ld");
  return 1;
}

