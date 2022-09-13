#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"
#include "../Specs/i16-128x128x4.h"

#ifndef U
#define U 8
#endif

void convertBitdepth(const TYPE* src, TYPE* dest) {
  #pragma ss config
  {
    arrayhint(src, N * C * sizeof(TYPE), 0);
    arrayhint(dest, N * C * sizeof(TYPE), 0);
    TYPE val;
    #pragma ss stream
    #pragma ss dfg dedicated unroll(U)
    for (int64_t j = 0; j < col_size * row_size * C; ++j) {
      TYPE t = src[j];
      t = max16(t, 0);
      t = min16(t, 255);
      dest[j] = t;
    }
  }
}

struct Arguments {
  TYPE src[N * C], dest[N * C];
  TYPE src_[N * C], dest_[N * C];
} args_;

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    convertBitdepth(args->src_, args->dest_);
  } else {
    convertBitdepth(args->src, args->dest);
  }
}

NO_INIT_DATA
NO_SANITY_CHECK
