// Test buffet reuse.

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#define N 32

struct Argument {
  int64_t a[N * N], b[N], c[N], ref[N];
} args_;


// FIXME(@were): This is not working because of reset/produce level analysis.
void buffet4d(int64_t *a, int64_t *b, int64_t *c) {
#pragma ss config
  {
#pragma ss stream
    for (int64_t i = 0; i < N; ++i) {
      int64_t acc = 0;
      for (int64_t j = 0; j < N; ++j) {
	for (int64_t ko = 0; ko < 2; ++ko) {
#pragma ss dfg dedicated
	  for (int64_t ki = 0; ki < N / 2; ++ki) {
	    int64_t k = ko * N / 2 + ki;
	    acc += a[i * N + k] * b[k];
	  }
	}
      }
      c[i] = acc;
    }
  }
}

struct Argument *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.b, N);
  return &args_;
}

void run_reference(struct Argument *args) {
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

void run_accelerator(struct Argument *args, int is_warmup) {
  buffet4d(args->a, args->b, args->c);
}

int sanity_check(struct Argument *args) {
  compare(args->c, args->ref, N, "%ld");
  return 1;
}

