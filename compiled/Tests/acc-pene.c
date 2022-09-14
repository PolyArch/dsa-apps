// Test of a single accumlator

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 10240
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N], b[N];
  int64_t ref;
  int64_t res;
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N);
  init_rand(args_.b, N);
  return &args_;
}

int64_t pene_sum(int64_t *a, int64_t *b) {
  int64_t res;
  #pragma ss config
  {
    int64_t acc = 0;
    #pragma ss stream
    #pragma ss dfg dedicated unroll(1)
    for (int i = 0; i < N; ++i)
      acc += a[b[i]];
    res = acc;
  }
  return res;
}

void run_reference(struct Arguments *args) {
  int64_t ref = 0;
  int64_t *a = args_.a;
  int64_t *b = args_.b;
  for (int i = 0; i < N; ++i)
    ref += a[b[i]];
  args_.ref = ref;
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  args->res = pene_sum(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  int cond = args->ref == args->res;
  if (!cond) {
    printf("ref: %ld != res: %ld\n", args->ref, args->res);
  }
  return cond;
}
