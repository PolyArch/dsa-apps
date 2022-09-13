/*
Implementation based on algorithm described in:
The cache performance and optimizations of blocked algorithms
M. D. Lam, E. E. Rothberg, and M. E. Wolf
ASPLOS 1991
*/

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"
#include "../Specs/nw.h"

#include <stdint.h>


struct Arguments {
  TYPE a[N], b[M], f[(N + 1) * (M + 1)];
  TYPE a_[N], b_[M], f_[(N + 1) * (M + 1)];
} args_;

NO_SANITY_CHECK
NO_INIT_DATA

void nw(TYPE *__restrict a, TYPE *__restrict b, TYPE *__restrict f){
  // for (int i = 0; i < N + 1; ++i) {
  //   f[i] = i * -1;
  // }
  // for (int i = 0; i < M + 1; ++i) {
  //   f[i * (N + 1)] = i * -1;
  // }
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 1; i <= M; ++i) {
      #pragma ss dfg
      for (int jo = 1; jo <= N; jo += 1) {
#define UNROLL_IMPL(j) \
        do {                                                \
          int64_t score = a[i] == b[j] ? 1 : -1;            \
          int64_t x = f[(i - 1) * (N + 1) + j - 1] + score; \
          int64_t y = f[(i - 1) * (N + 1) + j] - 1;         \
          int64_t z = f[i * (N + 1) + (j - 1)] - 1;         \
          f[i * (N + 1) + j] = max64(x, max64(x, y));       \
        } while (0)
        UNROLL_IMPL(jo + 0);
        // UNROLL_IMPL(jo + 1);
        // UNROLL_IMPL(jo + 2);
        // UNROLL_IMPL(jo + 3);
      }
    }
  }

}

void run_accelerator(struct Arguments *args, int warmup) {
  if (warmup) {
    nw(args->a_, args->b_, args->f_);
  } else {
    nw(args->a, args->b, args->f);
  }
}
