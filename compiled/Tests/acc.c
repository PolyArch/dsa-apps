// Test of a single accumlator
#include <stdint.h>
#include <stdlib.h>

#include "../Common/test.h"
#include "../Common/interface.h"

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
  int64_t *a = args_.a;
  init_linear(a, N);
  args_.ref = args_.res = 0;
  return &args_;
}

int64_t acc(int64_t a[N]) {
  int64_t res;
#pragma ss config
  {
    int64_t acc = 0;
#pragma ss stream
#pragma ss dfg dedicated unroll(2)
    for (int i = 0; i < N; ++i)
      acc += a[i] * a[i];
    res = acc;
  }
  return res;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *acc = &args->ref;
  for (int i = 0; i < N; ++i)
    (*acc) += a[i] * a[i];
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  args->res = acc(args->a);
}

int sanity_check(struct Arguments *args) {
  return args->res == args->ref;
}
