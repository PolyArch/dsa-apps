#include <stdint.h>

#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Common/spatial_inrin.h"
#undef FAKE
#include "../Specs/i16-128x128x4.h"

TYPE src[N], dest[N];

void bgr2grayscale(TYPE *dest, TYPE *src, TYPE w1, TYPE w2, TYPE w3, int cid){
  #pragma ss config
  {
    // The base workload each core should do.
    int64_t chunk = row_size / NUM_CORES;
    // The additional residues that some cores should do.
    int64_t residue = row_size % NUM_CORES;
    int64_t coef = 128;
    int64_t kilo = 128;
    // The starting address of this core (cid).
    const int64_t start = cid * chunk + mc_min(residue, cid);
    // Distribute the addtional residue.
    chunk += cid < residue;

    int64_t io, i, jo;
    #pragma ss stream 
    for(io = 0; io < chunk; ++io){
      #pragma ss dfg dedicated unroll (1) 
      for(jo = 0; jo < col_size; jo += 4){
        i = start + io;
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

void thread_entry(int cid, int nc) {
  int64_t coef = (72ll) | (715ll << 16) | (212ll << 32);
  int64_t kilo = (1000) | (1000ll << 16) | (1000ll << 32) | (1000ll << 48);
  barrier(nc);
  begin_roi();
  bgr2grayscale(dest, src, 72, 710, 210, cid);
  barrier(nc);
  end_roi();

  // return
#ifndef CHIPYARD
  sb_stats();
  if (cid != 0) {
    pthread_exit(NULL);
  }
#else
  exit(0);
#endif
}

