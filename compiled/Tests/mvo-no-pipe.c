// FIXME(@were): The correctness is not done.
//
// This is a trivial solution to mvo
// It introduces the problem of no preheader in the second sub-dfg

#include "../Common/test.h"
#include "../Common/timing.h"

#ifdef LONG_TEST
#define N 128
#else
#define N 32
#endif

struct Arguments {
  int64_t a[N * N], b[N], v[N], ref_a[N * N], ref_v[N];
} args_;

struct Arguments *init_data() {
  init_linear(args_.a, N * N);
  init_linear(args_.ref_a, N * N);
  init_linear(args_.b, N);
  init_linear(args_.v, N);
  init_linear(args_.ref_v, N);
  return &args_;
}

// reference, on CPU
void mvo_no_pipe_ref(int64_t *a, int64_t *b, int64_t *v) {
  {
    for (int i = 0; i < N; ++i) {
      int64_t acc = 0.0;
      for (int j = 0; j < N; ++j) {
	acc += a[i * N + j] * b[j];
      }
      v[i] = acc;
    }

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
	a[i * N + j] -= v[i] * b[j];
      }
    }
  }
}

void run_reference(struct Arguments *args) {
  mvo_no_pipe_ref(args->ref_a, args->b, args->ref_v);
}


// Accelerated, on DSA
void mvo_no_pipe(int64_t *a, int64_t *b, int64_t *v) {
#pragma ss config
  {
#pragma ss stream
    for (int i = 0; i < N; ++i) {
      int64_t acc = 0.0;
#pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j) {
	acc += a[i * N + j] * b[j];
      }
      v[i] = acc;
    }

#pragma ss stream
    for (int i = 0; i < N; ++i) {
#pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < N; ++j) {
	a[i * N + j] -= v[i] * b[j];
      }
    }
  }
}

void run_accelerator(struct Arguments *args, int iswarmup) {
  mvo_no_pipe(args->a, args->b, args->v);
  if (iswarmup) {
    init_linear(args_.a, N * N);
  }
}

int sanity_check(struct Arguments *args) {
  compare(args->v, args->ref_v, N, "%ld");
  compare(args->a, args->ref_a, N * N, "%ld");
  return 1;
}
