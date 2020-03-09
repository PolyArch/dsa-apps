// The first poc

#include <stdio.h>

#include "../Common/test.h"

#define N 32

double a[N];
double b[N];
double c[N];
double ref[N];

void kernel() {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

int main() {

  init<double>(a, N);
  init<double>(b, N);
  for (int i = 0; i < N; ++i) {
    ref[i] = a[i] + b[i];
  }

  kernel();
  begin_roi();
  kernel();
  end_roi();
  sb_stats();

  comp<double>(c, ref, N);
}
