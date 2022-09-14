/*
Implementation based on algorithm described in:
"Stencil computation optimization and auto-tuning on state-of-the-art multicore architectures"
K. Datta, M. Murphy, V. Volkov, S. Williams, J. Carter, L. Oliker, D. Patterson, J. Shalf, K. Yelick
SC 2008
*/

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Specs/stencil-3d.h"

#ifndef U
#define U -1
#endif

#ifndef NUM_CORES
#define NUM_CORES 4
#endif

TYPE a[SIZE], b[SIZE], C0, C1;

void stencil3d(TYPE orig[SIZE], TYPE sol[SIZE], TYPE C0, TYPE C1, int64_t cid) {

  // The base workload each core should do.
  int64_t chunk = (height_size - 2) / NUM_CORES;
  // The additional residues that some cores should do.
  int64_t residue = (height_size - 2) % NUM_CORES;
  // The starting address of this core (cid).
  int64_t start = cid * chunk + mc_min(residue, cid);
  // Distribute the addtional residue.
  chunk += cid < residue;


  // Stencil computation
  #pragma ss config
  {
    int64_t sum0, sum1, mul0, mul1;
    for(int64_t ii = 0; ii < chunk; ii++) {
      int64_t i = (start + ii + 1);
      for(int64_t j = 1; j < col_size - 1; j++) {
        #pragma ss stream nonblock
        #pragma ss dfg dedicated unroll(1)
        for(int64_t ko = 1; ko < row_size - 1; ko += 1) {
#define UNROLL_IMPL(k)                                         \
          sum0 = orig[INDX(row_size, col_size, k, j, i)];      \
          sum1 = orig[INDX(row_size, col_size, k, j, i + 1)] + \
                 orig[INDX(row_size, col_size, k, j, i - 1)] + \
                 orig[INDX(row_size, col_size, k, j + 1, i)] + \
                 orig[INDX(row_size, col_size, k, j - 1, i)] + \
                 orig[INDX(row_size, col_size, k + 1, j, i)] + \
                 orig[INDX(row_size, col_size, k - 1, j, i)];  \
          mul0 = sum0 * C0;                                    \
          mul1 = sum1 * C1;                                    \
          sol[INDX(row_size, col_size, k, j, i)] = mul0 + mul1;
          UNROLL_IMPL(ko + 0)
          // UNROLL_IMPL(ko + 1)
        }
      }
    }
  }
}

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  stencil3d(a, b, 123, 321, cid);
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

