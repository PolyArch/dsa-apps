#include "../Common/test.h"
#include "../Common/timing.h"

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
    for (int64_t i = 0; i < N; ++i) {
      #pragma ss stream
      for (int64_t j = 0; j < P / 4; ++j) {
        double acc[4] = {0, 0, 0, 0};
        #pragma ss dfg dedicated
        for (int64_t k = 0; k < M; ++k) {
          acc[0] += a[i * M + k] * b[k * P + (j * 4 + 0)];
          acc[1] += a[i * M + k] * b[k * P + (j * 4 + 1)];
          acc[2] += a[i * M + k] * b[k * P + (j * 4 + 2)];
          acc[3] += a[i * M + k] * b[k * P + (j * 4 + 3)];
        }
        c[i * P + (j * 4 + 0)] = acc[0];
        c[i * P + (j * 4 + 1)] = acc[1];
        c[i * P + (j * 4 + 2)] = acc[2];
        c[i * P + (j * 4 + 3)] = acc[3];
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
