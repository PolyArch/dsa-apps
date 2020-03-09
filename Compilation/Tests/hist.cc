// Test of atomic operation

#include <cstdio>
#include <cstring>
#include "../Common/test.h"

#ifndef N
#define N 8
#endif

int64_t a[N], b[N], hist[N], ref[N];

void kernel(int64_t *a, int64_t *hist) {
  #pragma ss config
  {
    int64_t l1_hist[N];
    #pragma ss stream
    #pragma ss dfg
    for (int i = 0; i < N; ++i) {
      l1_hist[a[i]] += 1;
    }

    //#pragma ss stream
    //#pragma ss dfg
    //for (int i = 0; i < N; ++i) {
    //  hist[i] = l1_hist[i];
    //}

  }
}

int main() {
  init_rand(a, N);

  for (int i = 0; i < N; ++i) {
    assert(a[i] < N);
    ref[a[i]] += 1;
  }

  kernel(a, hist);
  begin_roi();
  kernel(a, hist);
  end_roi();
  sb_stats();


  // TODO: For now this is only a sanity check.
  // comp(hist, ref, N);

  return 0;
}
