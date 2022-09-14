// Test of a single accumlator

#include <stdint.h>
#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N];
  int64_t ref;
  int64_t res;
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t ref = 0;
  for (int i = 0; i < N; ++i)
    ref = max64(ref, a[i]);
  args->ref = ref;
}

int64_t max_acc(int64_t *a) {
  int64_t res;
#pragma ss config
  {
    int64_t acc = 0;
#pragma ss stream
#pragma ss dfg dedicated unroll(4)
    for (int i = 0; i < N; ++i)
      acc = max64(acc, a[i]);
    res = acc;
  }
  return res;
}

void run_accelerator(struct Arguments *args) {
  args->res = max_acc(args->a);
}

int sanity_check(struct Arguments *args) {
  return args->ref == args->res;
}
