// Weighted addition

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"
#include "../Specs/i16-128x128x4-3x3.h"


struct Arguments {
  TYPE a[i_size], b[o_size];
  TYPE a_[i_size], b_[o_size];
} args_;

void blur(TYPE *__restrict a, TYPE *__restrict b) {
  int64_t coef = 9ll | (9ll << 16) | (9ll << 32) | (9ll << 48);
  #pragma ss config
  {
    arrayhint(a, i_size * sizeof(TYPE), -1);
    arrayhint(b, o_size * sizeof(TYPE), -1);
    #pragma ss stream
    for (int i = 0; i < o_row_size; ++i) {
      #pragma ss dfg dedicated unroll(1)
      for (int jo = 0; jo < o_col_size; jo += 2) {
        int64_t j = jo;
#define load_v(x,y) int64_t v##x##y = *((int64_t*) (a + (i + x) * i_col_size * C + (j + y) * C));
        load_v(0,0) load_v(0,1) load_v(0,2) load_v(0,3)
        load_v(1,0) load_v(1,1) load_v(1,2) load_v(1,3)
        load_v(2,0) load_v(2,1) load_v(2,2) load_v(2,3)
#define calc_mid(x) int64_t mid##x = add16x4(v##x##1, v##x##2);
        calc_mid(0)
        calc_mid(1)
        calc_mid(2)
        int64_t l0 = add16x4(v00, mid0);
        int64_t l1 = add16x4(v10, mid1);
        int64_t l2 = add16x4(v20, mid2);
        int64_t r0 = add16x4(v03, mid0);
        int64_t r1 = add16x4(v13, mid1);
        int64_t r2 = add16x4(v23, mid2);
        int64_t sum0 = add16x4(add16x4(l0, l1), l2);
        int64_t sum1 = add16x4(add16x4(r0, r1), r2);
        ((int64_t*)(b) + i * o_col_size + j)[0] = div16x4(sum0, coef);
        ((int64_t*)(b) + i * o_col_size + j)[1] = div16x4(sum1, coef);
      }
    }
  }
}

NO_INIT_DATA
NO_SANITY_CHECK

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    blur(args->a_, args->b_);
  } else {
    blur(args->a, args->b);
  }
}
