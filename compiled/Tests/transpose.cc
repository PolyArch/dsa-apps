// The first poc

#include <stdio.h>
#include <string.h>

#include "../Common/test.h"

#define N 16

double a[N * N];
double b[N * N];
double ref[N * N];

void kernel() {
  #pragma ss config
  {
    for (int i = 0; i < N; ++i) {
      #pragma ss stream
      #pragma ss dfg datamove
      for (int j = 0; j < N; ++j)
        b[i * N + j] = a[j * N + i];
    }
  }
}

int main() {

  init<double>(a, N * N);
  memset(b, 0, sizeof b);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i * N + j] = a[j * N + i];

  kernel();
  begin_roi();
  kernel();
  end_roi();
  sb_stats();

  comp<double>(ref, b, N);
}
