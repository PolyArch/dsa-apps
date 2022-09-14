// TODO(@were): Fix the correctness.

// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 36
#endif

struct Arguments {
  int64_t a[N], ref[N];
} args_;


void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *ref = args->ref;
  ref[0] = a[0];
  for (int i = 1; i < N; ++i) {
    ref[i] = ref[i - 1] + a[i];
  }
}

void prefix(int64_t *a) {
  #pragma ss config
  {
    int64_t acc = 0;
    #pragma ss stream
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i) {
      acc += a[i];
      a[i] = acc;
    }
  }
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  prefix(args->a);
  if (is_warmup) {
    init_linear(args->a, N);
  }
}

struct Arguments *init_data() {
  init_linear(args_.a, N);
  return &args_;
}

int sanity_check(struct Arguments *args) {
  compare(args->a, args->ref, N, "%ld");
  return 1;
}



