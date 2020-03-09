// Test of atomic operation

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "../Common/test.h"

#ifndef N
#define N 1024
#endif

#ifndef M
#define M 16384
#endif

#ifndef U
#define U 1
#endif

int64_t a[M], b[N];

void kernel(int64_t *a, int64_t *hist) {
  #pragma ss config
  {
    int64_t l1_hist[N];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(U)
    for (int i = 0; i < M; ++i) {
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

  for (int i = 0; i < M; ++i) {
    a[i] = rand() % N;
  }


  kernel(a, b);
  begin_roi();
  kernel(a, b);
  end_roi();
  sb_stats();


  // TODO: For now this is only a sanity check.
  // comp(hist, ref, N);

  return 0;
}
