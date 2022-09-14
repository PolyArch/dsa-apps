#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Specs/stencil-2d.h"

TYPE a[N * N], b[N * N], c[9];

void stencil2d(TYPE orig[N * N], TYPE sol[N * N], TYPE filter[9], int64_t cid) {
  // The base workload each core should do.
  int64_t chunk = (N - 2) / NUM_CORES;
  // The additional residues that some cores should do.
  int64_t residue = (N - 2) % NUM_CORES;
  // The starting address of this core (cid).
  int64_t start = (cid * chunk) + mc_min(residue, cid);
  // Distribute the addtional residue.
  chunk += cid < residue;

  #pragma ss config
  {
    int64_t ro, c, k1, k2;
    // TYPE s_filter[9];
    // TYPE s_orig[(chunk + 2) * N];
    // #pragma ss stream nonblock
    // for (int64_t ii = 0; ii < chunk + 2; ++ii) {
    //   int64_t i = start + ii;
    //   #pragma ss dfg dedicated unroll(4)
    //   for(int64_t j = 0; j < N; ++j) {
    //     s_orig[i * N + j] = orig[i * N + j];
    //   }
    // }

    // #pragma ss stream nonblock
    // #pragma ss dfg dedicated unroll(1)
    // for (int64_t i = 0; i < 9; ++i) {
    //   s_filter[i] = filter[i];
    // }

    for (ro = 0; ro < chunk; ro++) {
      for (k1 = 0; k1 < 3; k1++) {
        #pragma ss stream nonblock
        for (k2 = 0; k2 < 3; k2++) {
          #pragma ss dfg dedicated unroll(4)
          for (c = 0; c < N - 2; c++) {
            int64_t r = ro + start;
            TYPE mul = filter[k1 * 3 + k2] * orig[(r + k1) * N + k2 + c];
            sol[r * N + c] += mul;
          }
        }
      }
    }
  }
}

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  stencil2d(a, b, c, cid);
  barrier(nc);
  end_roi();
  sb_stats();
  #ifndef CHIPYARD
  sb_stats();
  if (cid != 0) {
    pthread_exit(NULL);
  }
  #else
  exit(0);
  #endif
}

