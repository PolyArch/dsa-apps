#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#define FAKE
#include "../Specs/i16-128x128x4.h"

#ifndef NUM_CORES
#define NUM_CORES 4
#endif

TYPE src[N], dest[N];

void channel_extract(TYPE *dest, TYPE *src, TYPE channel, int cid){
  #pragma ss config
  {
    // The base workload each core should do.
    int64_t chunk = row_size / NUM_CORES;
    // The additional residues that some cores should do.
    int64_t residue = row_size % NUM_CORES;
    // The starting address of this core (cid).
    int64_t start = cid * chunk + mc_min(residue, cid);
    #pragma ss stream
    for(int64_t ii = 0; ii < chunk; ++ii){
      #pragma ss dfg dedicated unroll(1)
      for(int64_t j = 0; j < col_size; ++j){
        int64_t i = ii + start;
        dest[i * col_size + j] = src[i * col_size * C + j * C + channel];
      }
    }
  }
}

void thread_entry(int cid, int nc) {

  barrier(nc);
  begin_roi();
  channel_extract(dest, src, 0, cid);
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

