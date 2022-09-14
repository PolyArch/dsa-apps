#include <stdint.h>

#include "../Common/test.h"
#include "../Common/timing.h"


#include <stdint.h>
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/i16-128x128x4-3x3.h"

TYPE a[i_size], b[o_size];

#define CHUNK (i_row_size / NUM_CORES)

void blur(TYPE *a, TYPE *b, int64_t cid){
  #pragma ss config
  {
    // The base workload each core should do.
    int64_t chunk = o_row_size / NUM_CORES;
    // The additional residues that some cores should do.
    int64_t residue = o_row_size % NUM_CORES;
    // The starting address of this core (cid).
    int64_t start = cid * chunk + mc_min(residue, cid);

    const int64_t coef = 9ll | (9ll << 16) | (9ll << 32) | (9ll << 48);
    for (int64_t io = 0; io < chunk; ++io) {
      #pragma ss stream
      #pragma ss dfg dedicated unroll(1)
      for (int64_t jo = 0; jo < o_col_size; ++jo) {
        int64_t i = start + io;
        int64_t j = jo;
#define load_v(x,y) int64_t v##x##y = *((int64_t*) (a + (i + x) * i_col_size * C + (j + y) * C));
        load_v(0,0) load_v(0,1) load_v(0,2)
        load_v(1,0) load_v(1,1) load_v(1,2)
        load_v(2,0) load_v(2,1) load_v(2,2)
#define calc_mid(x) int64_t mid##x = add16x4(v##x##1, v##x##2);
        calc_mid(0)
        calc_mid(1)
        calc_mid(2)
        int64_t l0 = add16x4(v00, mid0);
        int64_t l1 = add16x4(v10, mid1);
        int64_t l2 = add16x4(v20, mid2);
        int64_t sum0 = add16x4(add16x4(l0, l1), l2);
        ((int64_t*)(b + i * o_col_size + j))[0] = div16x4(sum0, coef);
      }
    }
  }
}

void thread_entry(int cid, int nc) {
  // run
  barrier(nc);
  begin_roi();
  blur(a, b, cid);
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

