#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Specs/solver.h"


void solver(TYPE a[N * N], TYPE v[N], int64_t cid) {
  if (cid) return;
  #pragma ss config
  {
    for (int i = 0; i < N - 1; ++i) {
      TYPE vv = 0;
      TYPE v0 = v[i];
      TYPE a0 = a[i * N + i];
      #pragma ss dfg temporal
      {
        vv = v0 / a0;
      }
      // v[i] = vv;
      #pragma ss stream nonblock
      #pragma ss dfg dedicated unroll(4)
      for (int j = i + 1; j < N; ++j) {
        v[j] -= a[i * N + j] * vv;
      }
    }
  }
}

TYPE a[N * N], v[N];

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  solver(a, v, cid);
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

