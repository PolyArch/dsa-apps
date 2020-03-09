#include "../Common/test.h"

#ifndef N
#define N 130
#endif

#ifndef U
#define U 4
#endif

void stencil (int64_t orig[N * N], int64_t sol[N * N], int64_t filter[9]){

  #pragma ss config
  {
    int r, c, k1, k2;
    int64_t temp, mul;
    for (r = 0; r < N - 2; r++) {
        temp = (int64_t)0;
        for (k1 = 0; k1 < 3; k1++){                    //Row access
          #pragma ss stream nonblock
          for (k2 = 0; k2 < 3; k2++){                //column access
            #pragma ss dfg dedicated unroll(U)
            for (c = 0; c < N - 2; c++) {
              mul = filter[k1 * 3 + k2] * orig[(r + k1) * N + c + k2];
              sol[r * N + c] += mul;
            }
        }
      }
    }
  }
}

int64_t a[N * N], b[N * N], c[9];

int main() {
  stencil(a, b, c);
  begin_roi();
  stencil(a, b, c);
  end_roi();
  sb_stats();
  return 0;
}
