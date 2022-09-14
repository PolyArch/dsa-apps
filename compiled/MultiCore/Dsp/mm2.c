#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/mm.h"

#ifndef U
#define U 4
#endif

#define CHUNK (N / NUM_CORES)

TYPE a[N * M], b[M * P], c[N * P];

void mm2(TYPE *a, TYPE *b, TYPE *c, int cid){

  // The base workload each core should do.
  int64_t chunk = N / NUM_CORES;
  // The additional residues that some cores should do.
  int64_t residue = N % NUM_CORES;
  // The starting address of this core (cid).
  int64_t start = cid * chunk + mc_min(residue, cid);
  // Distribute the addtional residue.
  chunk += cid < residue;

  #pragma ss config
  {
    TYPE bb[M * P];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int64_t i = 0; i < M * P; ++i)
      bb[i] = b[i];
    for (int64_t ii = 0; ii < chunk; ++ii) {
      int64_t i = ii + start;
      #pragma ss stream nonblock
      for (int64_t k = 0; k < M; ++k) {
        #pragma ss dfg dedicated unroll(4)
        for (int64_t j = 0; j < P; ++j) {
          c[i * P + j] += a[i * M + k] * bb[k * P + j];
        }
      }
    }
  }
}

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  mm2(a, b, c, cid);
  barrier(nc);
  end_roi();

#ifndef CHIPYARD
  sb_stats();
  if (cid != 0) {
    pthread_exit(NULL);
  }
#else
  exit(0);
#endif
}

