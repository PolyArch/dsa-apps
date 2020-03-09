// Test of a single accumlator

#include <algorithm>
#include <stdio.h>

#include "../Common/test.h"

#define N 128

#ifndef U
#define U 4
#endif

double a[N];

double kernel(double *a) {
  double res;
  #pragma ss config
  {
    double acc = 0.0;
    #pragma ss stream
    #pragma ss dfg dedicated unroll(U)
    for (int i = 0; i < N; ++i)
      acc += a[i] * a[i];
    res = acc;
  }
  return res;
}

int main() {

  init<double>(a, N);
  double ref = 0.0;
  for (int i = 0; i < N; ++i)
    ref += a[i] * a[i];

  kernel(a);
  begin_roi();
  double ans = kernel(a);
  end_roi();
  sb_stats();

  assert(std::abs(ans - ref) < 1e-5);
  return 0;
}
