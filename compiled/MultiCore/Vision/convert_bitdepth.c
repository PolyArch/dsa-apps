#include <stdint.h>

#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/i16-128x128x4.h"

#ifndef U

#ifdef FAKE
#define U 4
#else
#define U 16
#endif

#endif

TYPE src[N], dest[N];

void convert_bitdepth(TYPE *t, TYPE *dest, int cid){
  #pragma ss config
  {
    // The base workload each core should do.
    int64_t chunk = row_size / NUM_CORES;
    // The additional residues that some cores should do.
    int64_t residue = row_size % NUM_CORES;
    // The starting address of this core (cid).
    int64_t start = cid * chunk + mc_min(residue, cid);
    // Distribute the addtional residue.
    chunk += cid < residue;
    chunk *= col_size * C;
    start *= col_size * C;
    #pragma ss stream
    #pragma ss dfg dedicated unroll(U)
    for (int64_t jj = 0; jj < chunk; ++jj) {
      int64_t j = start + jj;
      TYPE t = src[j];
#ifdef FAKE
      t = max64(t, 0);
      t = min64(t, 255);
#else
      t = max16(t, 0);
      t = min16(t, 255);
#endif
      dest[j] = t;
    }
  }
}

void thread_entry(int cid, int nc) {
  // run
  barrier(nc);
  begin_roi();
  convert_bitdepth(src, dest, cid);
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

