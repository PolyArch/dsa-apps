// The first proof of concept

#include <stdio.h>

#include "common/test.h"
#include "common/timing.h"
#include "common/multicore.h"

#define N 1024

#ifndef TYPE
#define TYPE int64_t
#endif

void vecadd(TYPE *a, TYPE *b, TYPE *c, int64_t cid) {
  #pragma ss config
  {

    // The base workload each core should do.
    int64_t chunk = N / NUM_CORES;
    // The additional residues that some cores should do.
    int64_t residue = N % NUM_CORES;
    // The starting address of this core (cid).
    int64_t start = cid * chunk + mc_min(residue, cid);

    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int i = 0; i < chunk; ++i) {
      c[i + start] = a[i + start] + b[i + start];
    }
  }
}

TYPE a[N], b[N], c[N];

void thread_entry(int cid, int nc) {
  // run
  barrier(nc);
  begin_roi();
  vecadd(a, b, c, cid);
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

