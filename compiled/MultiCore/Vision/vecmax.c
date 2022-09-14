// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/multicore.h"
#include "../Specs/i16-128x128x4.h"

TYPE a[N], b[N], c[N];

#ifndef U

#ifdef FAKE
#define U 4
#else
#define U 16
#endif

#endif

void vecmax(TYPE *a, TYPE *b, TYPE *c, int cid){

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

  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated unroll(U)
    for (int64_t ii = 0; ii < chunk; ++ii) {
      int64_t i = ii + start;
#ifdef FAKE
      c[i] = max64(a[i], b[i]);
#else
      c[i] = max16(a[i], b[i]);
#endif
    }
  }  
}

void thread_entry(int cid, int nc) {

  barrier(nc);
  begin_roi();
  vecmax(a, b, c, cid);
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
