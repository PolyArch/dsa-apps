#include <cstdio>
#include "solver.h"

#define N _N_

complex<float> a[N * N], v[N];

int main() {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j <= i; ++j)
      scanf("%f", a + i * N + j);
  for (int i = 0; i < N; ++i)
    scanf("%f", v + i);
  solver(a, v);
  return 0;
}

#undef N
