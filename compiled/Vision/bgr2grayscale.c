#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"
#include "../Specs/i16-128x128x4.h"

void bgr2grayscale(const TYPE* src, TYPE* dest){
  #pragma ss config
  {
    int64_t coef = (72ll) | (715ll << 16) | (212ll << 32);
    int64_t kilo = (1000) | (1000ll << 16) | (1000ll << 32) | (1000ll << 48);
    arrayhint(src, N * sizeof(TYPE), -1);
    arrayhint(dest, N * sizeof(TYPE), -1);
    int64_t i, jo;
    #pragma ss stream 
    for(i = 0; i < row_size; ++i) {
      #pragma ss dfg dedicated unroll(1) 
      for(jo = 0; jo < col_size; jo += 4) {
        int64_t j = jo;
        int64_t indexSrc = i * col_size + j;
        int64_t indexDest = i * col_size + j;
        int64_t indexColor = i * col_size + j * C;
        int64_t v0 = ((int64_t*)(src + indexColor))[0];
        int64_t v1 = ((int64_t*)(src + indexColor))[1];
        int64_t v2 = ((int64_t*)(src + indexColor))[2];
        int64_t v3 = ((int64_t*)(src + indexColor))[3];
        int64_t m0 = hladd64(hladd32x2(mul16x4(v0, coef)));
        int64_t m1 = hladd64(hladd32x2(mul16x4(v1, coef)));
        int64_t m2 = hladd64(hladd32x2(mul16x4(v2, coef)));
        int64_t m3 = hladd64(hladd32x2(mul16x4(v3, coef)));
        int64_t m4 = concat64(concat32x2(m0, m1), concat32x2(m2, m3));
        ((int64_t*)(dest + indexDest))[0] = div16x4(m4, kilo);
      }
    }
  }
}

struct Arguments {
  TYPE image[N * C], dest[N];
  TYPE image_[N * C], dest_[N];
} args_;

NO_INIT_DATA
NO_SANITY_CHECK

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    bgr2grayscale(args->image_, args->dest_);
  } else {
    bgr2grayscale(args->image, args->dest);
  }
}
