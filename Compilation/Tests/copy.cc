// Test on data copy, the memcpy intrin.

#include <stdio.h>

#include "../Common/test.h"

#define N 32

double a[N];
double b[N];
double ref[N];

void kernel() {
  #pragma ss config
  {

    #pragma ss stream
    #pragma ss dfg
    for (int i = 0; i < N; ++i) {
      b[i] = a[i];
    }

  }
}

int main() {

  init<double>(a, N);
  init<double>(ref, N);

  begin_roi();
  kernel();
  end_roi();
  sb_stats();

  comp<double>(b, ref, N);
}
