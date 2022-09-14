#include <stdint.h>

#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Specs/i16-128x128x4.h"

#ifndef NUM_CORES
#define NUM_CORES 4
#endif

#ifndef U
#define U 4
#endif

TYPE src[N], dest[N];

void accumulate(TYPE *dest, TYPE *src, int64_t cid){
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

    #pragma ss stream nonblock
    #pragma ss dfg dedicated unroll(U)
    for (int64_t i = 0; i < chunk; ++i){
      dest[start + i] += src[start + i];
    }
  }
}

void thread_entry(int cid, int nc) {
  // run
  barrier(nc);
  begin_roi();
  accumulate(dest, src, cid);
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

