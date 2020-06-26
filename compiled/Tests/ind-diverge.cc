// Test of indirect memory access. The indirect address diverges.

#include <cstdio>
#include "../Common/test.h"

#ifndef N
#define N 32
#endif

int a[N];
int64_t b[N], c[N], d[N], ref[N];

void kernel(int *a, int64_t *b, int64_t *c, int64_t *d) {
  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg
    for (int i = 0; i < N; ++i) {
      d[i] = b[a[i]] + c[a[i]];
    }
  }
}

int main() {
  init_rand(a, N);
  init_rand(b, N);
  init(c, N);

  for (int i = 0; i < N; ++i) {
    ref[i] = b[a[i]] + c[a[i]];
  }

  kernel(a, b, c, d);
  begin_roi();
  kernel(a, b, c, d);
  end_roi();
  sb_stats();


  comp<int64_t>(d, ref, N);

  return 0;
}
