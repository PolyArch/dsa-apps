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

#define CHUNK (N / NUM_CORES)


TYPE src[N], dest[N];

void accumulate_squared(TYPE *dest, TYPE *src, int cid){
  #pragma ss config
  {
    int64_t chunk = row_size / NUM_CORES;
    int64_t residue = row_size % NUM_CORES;
    int64_t start = cid * chunk + mc_min(residue, cid);
    chunk += cid < residue;
    chunk *= col_size * C;
    start *= col_size * C;

    #pragma ss stream nonblock
    #pragma ss dfg dedicated unroll(U)
    for (int64_t i = 0; i < chunk; ++i){
      dest[i + start] += src[i + start] * src[i + start];
    }
  }
}

void thread_entry(int cid, int nc) {
  // run
  barrier(nc);
  begin_roi();
  accumulate_squared(dest, src, cid);
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

