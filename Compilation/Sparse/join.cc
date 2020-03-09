#include <cstdio>
#include "../Common/test.h"
#include "../Common/utils.h"

#ifndef N
#define N 1024
#endif

#ifndef BATCH
#define BATCH 2
#endif

uint64_t atomic[N + 1];
uint64_t idx0[BATCH][N + 1];
uint64_t v0[BATCH][N + 1];
uint64_t idx1[BATCH][N + 1];
uint64_t v1[BATCH][N + 1];
uint64_t sum[BATCH][1];

#define n0 (N * 3 / 4)
#define n1 (N * 6 / 7)

#ifndef U
#define U 1
#endif

void kernel(uint64_t idx0[][N + 1],
            uint64_t v0[][N + 1],
            uint64_t idx1[][N + 1],
            uint64_t v1[][N + 1],
            uint64_t sum[][1]) {
  #pragma ss config
  {
    for (int64_t j = 0; j < BATCH; ++j) {
      uint64_t acc = 0;
      #pragma ss stream nonblock
      #pragma ss dfg dedicated unroll(U)
      for (int64_t i0 = 0, i1 = 0; i0 < n0 && i1 < n1; ) {
        if (idx0[j][i0] == idx1[j][i1]) {
          acc += v0[j][i0] * v1[j][i1];
          ++i0;
          ++i1;
        } else {
          idx0[j][i0] < idx1[j][i1] ? ++i0 : ++i1;
        }
      }
      sum[j][0] = acc;
    }
  }
}

int main() {

  for (int i = 0; i < N; ++i) {
    atomic[i] = i;
  }

  //int64_t n0 = N * 3 / 4;
  //int64_t n1 = N * 6 / 7;
  random_shuffle(N, atomic);
  merge_sort(n0, atomic);
  for (int j = 0; j < BATCH; ++j) {
    for (int i = 0; i < n0; ++i) {
      idx0[j][i] = atomic[i];
    }
    idx0[j][n0] = 1ull << 63;
  }

  for (int j = 0; j < BATCH; ++j) {
    random_shuffle(N, atomic);
    merge_sort(n1, atomic);
    for (int i = 0; i < n1; ++i) {
      idx1[j][i] = atomic[i];
    }
    idx1[j][n1] = 1ull << 63;
  }


  kernel(idx0, v0, idx1, v1, sum);
  begin_roi();
  kernel(idx0, v0, idx1, v1, sum);
  end_roi();
  sb_stats();

  assert(sum[0][0] == 0);

  return 0;
}
