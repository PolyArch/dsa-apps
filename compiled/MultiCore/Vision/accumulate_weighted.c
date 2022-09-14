#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/i16-128x128x4.h"

#ifndef U
#define U 4
#endif

TYPE src[N], dest[N];

void accumulate_weighted(TYPE *src, TYPE *dest, int cid){
  #pragma ss config
  {
    // The base workload each core should do.
    int64_t chunk = row_size / NUM_CORES;
    // The additional residues that some cores should do.
    int64_t residue = row_size % NUM_CORES;
    // The starting address of this core (cid).
    int64_t start = cid * chunk + mc_min(residue, cid);
    TYPE temp1 = 456;
    TYPE temp2 = 345;
    TYPE twoPower23 = 123;
    // Distribute the addtional residue.
    chunk += cid < residue;
    chunk *= col_size * C;
    start *= col_size * C;

    #pragma ss stream
    #pragma ss dfg dedicated unroll(U)
    for (int64_t i = 0; i < chunk; ++i){
#ifndef FAKE
      dest[start + i] = div16(src[start + i] * temp1 + dest[start + i] * temp2,  twoPower23);
#else
      dest[start + i] = src[start + i] * temp1 + dest[start + i] * temp2 / twoPower23;
#endif
    }
  }
}

void thread_entry(int cid, int nc) {
  // run
  barrier(nc);
  begin_roi();
  accumulate_weighted(src, dest, cid);
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

