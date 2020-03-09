#include <cstdio>
#include "../Common/test.h"

#define N 8

uint64_t idx0[N / 2 + 1] = {2, 3, 5, 11, 1ull << 63};
uint64_t v0[N / 2 + 1]       = {1, 2, 3, 4, 0};
uint64_t idx1[N + 1]     = {1, 2, 3, 4, 5, 6, 7, 8, 1ull << 63};
uint64_t v1[N + 1]           = {1, 2, 3, 4, 5, 6, 7, 8, 0};

int main() {

  int64_t sum = 0;

  begin_roi();
  #pragma ss config
  {
    int64_t acc = 0;
    #pragma ss stream
    #pragma ss dfg
    for (int i0 = 0, i1 = 0; i0 < N / 2 && i1 < N; ) {
      if (idx0[i0] == idx1[i1]) {
        acc += v0[i0] * v1[i1];
        ++i0;
        ++i1;
      } else {
        idx0[i0] < idx1[i1] ? ++i0 : ++i1;
      }
    }
    sum = acc;
  }
  end_roi();
  sb_stats();

  assert(sum == 23);
  return 0;
}
