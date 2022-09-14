#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"

#ifndef N
#define N 32
#endif

#ifndef M
#define M 32
#endif

#ifndef P
#define P 32
#endif

#ifndef U
#define U 4
#endif

void mm(double *a, double *b, double *c) {
  #pragma ss config
  {
    for (int i = 0; i < N; ++i) {
      #pragma ss stream nonblock
      for (int k = 0; k < M; ++k) {
        #pragma ss dfg dedicated unroll(-1)
        for (int j = 0; j < P; ++j) {
          c[i * P + j] += a[i * M + k] * b[k * P + j];
        }
      }
    }
  }
}

struct Arguments {
  double a[N * M], b[M * P], c[N * P];
} args_;

NO_INIT_DATA
NO_SANITY_CHECK

void run_accelerator(struct Arguments *args, int _) {
  mm(args->a, args->b, args->c);
}
