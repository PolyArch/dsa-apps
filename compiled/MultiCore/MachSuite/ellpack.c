#include <stdint.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Specs/ellpack.h"

#include <stdint.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"

#define CHUNK (N / NUM_CORES)


TYPE val[N * L], vec[N], out[N];
int64_t cols[N * L];

void ellpack(TYPE *val, TYPE *vec, TYPE *out, int cid) {

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
    TYPE spad[N];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int64_t i = 0; i < N; ++i) {
      spad[i] = vec[i];
    }
    #pragma ss stream
    for (int64_t ii = 0; ii < chunk; ++ii) {
      int64_t i = ii + start;
      TYPE sum = 0.0;
      #pragma ss dfg dedicated unroll(4)
      for (int64_t j = 0; j < L; ++j) {
        sum += val[j + i * L] * spad[cols[j + i * L]];
      }
      out[i] = sum;
    }
  }
}

void thread_entry(int cid, int nc) {
  // The base workload each core should do.
  int64_t chunk = N / NUM_CORES;
  // The additional residues that some cores should do.
  int64_t residue = N % NUM_CORES;
  // The starting address of this core (cid).
  int64_t start = cid * chunk + mc_min(residue, cid);
  // Distribute the addtional residue.
  chunk += cid < residue;

  {
    for (int ii = 0; ii < chunk; ++ii) {
      int i = ii + start;
      for (int j = 0; j < L; ++j) {
        val[i * L + j] = rand();
        cols[i * L + j] = rand() % N;
      }
    }
  }

  barrier(nc);
  begin_roi();
  ellpack(val, vec, out, cid);  
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
