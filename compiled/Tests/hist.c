// FIXME(@were): Implement a 0-initializer.
// FIXME(@were): Fix this. The atomic codegen may be wrong.

// Test of atomic operation

#include <string.h>

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 1024
#else
#define N 8
#endif

struct Arguments {
  int64_t a[N], b[N], h[N], ref[N];
} args_;

struct Arguments *init_data() {
  // data initialization
  init_rand(args_.a, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  int64_t *a = args->a;
  int64_t *b = args->ref;
  // calculate reference
  unsigned long start = rdcycle();
  for (int i = 0; i < N; ++i) {
    b[a[i]] += 1;
  }
}

void hist(int64_t *a, int64_t *hist) {
  #pragma ss config
  {
    int64_t l1_hist[N];
    #pragma ss stream
    #pragma ss dfg
    for (int i = 0; i < N; ++i) {
      l1_hist[a[i]] += 1;
    }
    #pragma ss stream
    #pragma ss dfg
    for (int i = 0; i < N; ++i) {
      hist[i] = l1_hist[i];
    }
  }
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  hist(args->a, args->h);
}

int sanity_check(struct Arguments *args) {
  for (int i = 0; i < N; ++i) {
    args->h[i] /= 2;
  }
  compare(args->h, args->ref, N, "%ld");
  return 1;
}

