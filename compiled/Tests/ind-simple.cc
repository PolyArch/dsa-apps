// Test simple indirect memory access

#include <cstdio>
#include "../Common/test.h"

#ifndef N
#define N 32
#endif

int a[N], b[N];
int64_t c[N], d[N], ref[N];

void kernel(int *a, int *b, int64_t *c, int64_t *d) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg unroll(4)
    for (int i = 0; i < N; ++i) {
      d[i] = c[a[i]] + c[b[i]];
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
