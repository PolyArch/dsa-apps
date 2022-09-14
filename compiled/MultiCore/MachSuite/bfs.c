/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include <stdint.h>
#include <string.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/multicore.h"
#include "../Specs/bfs.h"

#ifndef U
#define U 1
#endif

void bfs(int64_t source, int64_t h[N], int64_t e[M], int64_t begin[N], int64_t end[N], int64_t cid) {

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
    if (cid == 0) {
      h[source] = 0;
    }
    barrier(NUM_CORES);
    int64_t flag = 1;
    int64_t curh = 0;
    while (curh < 7) {
      flag = 0;
      int64_t res = curh + 1;
      for (int64_t ii = 0; ii < chunk; ++ii) {
        int64_t i = ii + start;
        if (h[i] == curh) {
          flag = 1;
          int64_t l = begin[i], r = end[i];
          #pragma ss stream
          #pragma ss dfg
          for (int j = l; j < r; ++j) {
            int64_t *ptr = h + e[j];
            *ptr = min64(*ptr, res);
          }
        }
      }
      curh = curh + 1;
      barrier(NUM_CORES);
    }
  }
}

int64_t source;
int64_t h[N];
int64_t e[M];
int64_t begin[N];
int64_t end[N];


void thread_entry(int cid, int nc) {
  int64_t total = 0;
  if (cid == 0) {
    for (int64_t i = 0; i < N; ++i) {
      h[i] = N + 1;
      begin[i] = total;
      total += rand() % E + 1;
      end[i] = total;
      for (int64_t j = begin[i]; j < end[i]; ++j) {
        e[j] = rand() % N;
      }
    }
  }
  source = rand() % N;
  barrier(nc);

  begin_roi();
  bfs(source, h, e, begin, end, cid);
  barrier(nc);
  end_roi();
  sb_stats();

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

