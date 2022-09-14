// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"

#define M 3
#define N 4
#define P 5
#define K 6

#ifndef TYPE
#define TYPE int64_t
#endif

#if TYPE == int64_t
#define TY_FMT "%ld"
#elif TYPE == int32_t
#define TY_FMT "%d"
#elif TYPE == double || TYPE == float
#define TY_FMT "%f"
#endif

struct Arguments {
  TYPE a[M*N*P*K], b[M*N*P*K], c[M*N*P*K], ref[M*N*P*K];
} args_;

void run_reference(struct Arguments *args) {
  TYPE *a = args->a;
  TYPE *b = args->b;
  TYPE *c = args->ref;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < P; ++k) {
        for (int l = 0; l < K; ++l) {
          c[i*N*P*K+j*P*K+k*K+l] = a[i*M*P*K+j*N*P*K+k*M*N*P+l] * b[j];
        }
      }
    }
  }
}

void fourdadd(TYPE *a, TYPE *b, TYPE *c) {
  #pragma ss config
  {
    // arrayhint(a, N * sizeof(TYPE), 0);
    // arrayhint(b, N * sizeof(TYPE), 0);
    // arrayhint(c, N * sizeof(TYPE), 0);
    #pragma ss stream
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < P; ++k) {
          #pragma ss dfg dedicated
          for (int l = 0; l < K; ++l) {
            c[i*N*P*K+j*P*K+k*K+l] = a[i*M*P*K+j*N*P*K+k*M*N*P+l] * b[j];
          }
        }
      }
    }
  }
}

void run_accelerator(struct Arguments *args) {
  fourdadd(args->a, args->b, args->c);
}

int sanity_check(struct Arguments *args) {
  compare(args->c, args->ref, N, TY_FMT);
  return 1;
}

struct Arguments *init_data() {
  // data initialization
  init_odd(args_.a, N);
  init_even(args_.b, N);
  return &args_;
}

