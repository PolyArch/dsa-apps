#include "../Common/test.h"

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

void kernel(double *a, double *b, double *c) {
  #pragma ss config
  {
    //double bb[M * P];
    //for (int i = 0; i < M * P; ++i)
    //  bb[i] = b[i];
    for (int i = 0; i < N; ++i)
      #pragma ss stream nonblock
      for (int k = 0; k < M; ++k) {
        #pragma ss dfg dedicated unroll(U)
        for (int j = 0; j < P; ++j) {
          //c[i * P + j] += a[i * M + k] * bb[k * P + j];
          c[i * P + j] += a[i * M + k] * b[k * P + j];
        }
      }
  }
}

double a[N * M], b[M * P], c[N * P];

int main() {

  kernel(a, b, c);
  begin_roi();
  kernel(a, b, c);
  end_roi();
  sb_stats();
}
