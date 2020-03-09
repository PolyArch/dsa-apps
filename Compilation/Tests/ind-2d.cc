// Test simple indirect memory access

#include <cstdio>
#include "../Common/test.h"

#ifndef N
#define n 4
#define N (n * n)
#endif

int a[N], b[N];
int64_t c[N], d[N], ref[N];

void kernel(int *a, int *b, int64_t *c, int64_t *d) {
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < n; ++i) {
      #pragma ss dfg unroll(4)
      for (int j = 0; j < n; ++j) {
        d[i * n + j] = c[a[i * n + j]] + c[b[i * n + j]];
      }
    }
  }
}

int main() {
  init_rand(a, N);
  init_rand(b, N);
  init(c, N);

  for (int i = 0; i < N; ++i) {
    ref[i] = c[a[i]] + c[b[i]];
  }

  kernel(a, b, c, d);
  begin_roi();
  kernel(a, b, c, d);
  end_roi();
  sb_stats();


  comp<int64_t>(d, ref, N);

  return 0;
}
