// Test of communication between temporal and dedicated region

#include <stdio.h>
#include <math.h>

#include "../Common/test.h"

#define N 16

double a[N];
double b[N];

int main() {
  init<double>(a, N);
  init<double>(b, N);

  double len = 0.0;
  for (int i = 0; i < N; ++i)
    len += a[i] * a[i];
  len = sqrt(1.0 / len);
  for (int i = 0; i < N; ++i)
    a[i] *= len;

  begin_roi();
  #pragma ss config
  {
    double acc = 0.0;
    #pragma ss stream nonblock
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i)
      acc += b[i] * b[i];

    #pragma ss dfg temporal
    {
      acc = sqrt(1.0 / acc);
    }

    #pragma ss stream
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i)
      b[i] *= acc;
  }
  end_roi();
  sb_stats();

  comp<double>(a, b, N);
  return 0;
}
