
#include "../Common/timing.h"
#include "../Common/test.h"

#ifndef N
#define N 32
#endif

#ifndef U
#define U 2
#endif


void qr_r(double *a, double *q, double *tau) {

  #pragma ss config
  {
    double w[N];
    for (int64_t i = N - 2; i >= 0; --i) {
      int n = N - i;

      #pragma ss stream
      #pragma ss dfg dedicated unroll(-1)
      for (int64_t j = i + 1; j < N; ++j)
        w[j - i] = a[j * N + i];

      #pragma ss stream
      for (int64_t j = i; j < N; ++j) {
        double acc = 0.0;
        #pragma ss dfg dedicated unroll(-1)
        for (int64_t k = i; k < N; ++k)
          acc += q[j * N + k];
        #pragma ss dfg dedicated unroll(-1)
        for (int64_t k = i; k < N; ++k)
          q[j * N + k] -= tau[i] * w[k - i] * acc;
      }
    }
  }
}

double a[N * N], q[N * N], tau[N];
double aa[N * N], qq[N * N], tt[N];

int main() {
  for (int i = 0; i < N * N; ++i)
    a[i] = i + 1;
  qr_r(aa, qq, tt);
  begin_roi();
  qr_r(a, q, tau);
  end_roi();
  sb_stats();
  return 0;
}
