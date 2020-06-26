#include <cstdint>

#include "../Common/test.h"

#ifndef N
#define N 494
#endif

#ifndef L
#define L 16
#endif

#ifndef U
#define U 1
#endif

void ellpack(int64_t *nzval, int64_t *cols, int64_t *vec, int64_t *out) {
  #pragma ss config
  {
    #pragma ss stream nonblock
    for (int i = 0; i < N; ++i) {
      int64_t sum = 0.0;
      #pragma ss dfg dedicated unroll(U)
      for (int j = 0; j < L; ++j) {
        sum += nzval[j + i * L] * vec[cols[j + i * L]];
      }
      out[i] = sum;
    }
  }
}

int64_t val[N * L], vec[N], out[N];
int64_t cols[N * L];

int main() {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < L; ++j) {
      val[i * L + j] = rand();
      cols[i * L + j] = rand() % N;
    }

  ellpack(val, cols, vec, out);
  begin_roi();
  ellpack(val, cols, vec, out);
  end_roi();
  sb_stats();

  return 0;
}
