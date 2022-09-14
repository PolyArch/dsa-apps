#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/cholesky.h"

TYPE a[(N + NUM_CORES) * N], L[N * N];

#ifndef U
#define U 4
#endif

void cholesky(TYPE *a, TYPE one, int64_t cid){
#pragma ss config
  {
    for (int64_t i = 0; i < N - 2; ++i) {
      int64_t n = N - (i + 1);
      int64_t chunk = (n - 1) / NUM_CORES;
      int64_t residue = (n - 1) % NUM_CORES;
      int64_t start = cid * chunk;
      start += mc_min(cid, residue) - (cid != 0);
      chunk += cid && cid <= residue;

      TYPE sqrt_inv, inv, aii = a[i * (N + 1)];
      if (cid == 0) {
#pragma ss dfg temporal
      {
	sqrt_inv = one / fsqrt(aii);
      }
#pragma ss stream nonblock
#pragma ss dfg dedicated
	for (int64_t j = i; j < N; ++j)
	  L[j * N + i] = a[i * N + j + 1] * sqrt_inv;
      }

      // printf("%d, %d: %d %d\n", (int) i, (int) cid, (int) start, (int) chunk);
      #pragma ss stream
      for (int64_t jo = 0; jo < chunk; ++jo) {
        int64_t j = jo + start;
        #pragma ss dfg dedicated unroll(2)
        for (int64_t k = j; k < N; ++k) {
          a[j * N + k] -= a[i * N + j] * a[i * N + k + 1]; // * inv;
        }
      }
      barrier(NUM_CORES);
    }
  }
}

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  cholesky(a, 1.0, cid);
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

