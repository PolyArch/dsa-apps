#include <stdint.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Specs/crs.h"

#define CHUNK (N / NUM_CORES)


TYPE val[N * 4], out[N], vec[N];
int64_t tots[NUM_CORES], ns[N], begin[N + 1], col[N * 4];

void crs(TYPE *val, TYPE *vec, TYPE *out, int64_t *begin, int64_t *ns, int cid){
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
    #pragma ss dfg dedicated
    for (int64_t i = 0; i < N; ++i) {
      spad[i] = vec[i];
    }

    int64_t k = 0;
    #pragma ss stream nonblock fifo(col, tots[cid])
    for (int64_t ii = 0; ii < chunk; ++ii) {
      int64_t i = ii + start;
      TYPE sum = 0;
      #pragma ss dfg dedicated unroll(1)
      for (int64_t j = 0; j < ns[i]; ++j){
        sum += val[begin[i] + j] * spad[col[k]];
        ++k;
      }
      out[i] = sum;
    }
  }
}

void thread_entry(int cid, int nc) {
  int64_t total = 0;
  {

    // The base workload each core should do.
    int64_t chunk = N / NUM_CORES;
    // The additional residues that some cores should do.
    int64_t residue = N % NUM_CORES;
    // The starting address of this core (cid).
    int64_t start = cid * chunk + mc_min(residue, cid);
    // Distribute the addtional residue.
    chunk += cid < residue;

    // init_linear((vec + cid * CHUNK), CHUNK);
    total = cid * CHUNK * 4;
    for (int ii = 0; ii < chunk; ++ii) {
      int i = ii + start;
      vec[i] = i + 1;
      begin[i] = total;
      int n = 3 + (rand() % 3 == 0);
      ns[i] = n;
      for (int j = total; j < total + n; ++j) {
        val[j] = rand();
        col[j] = rand() % N;
      }
      total += n;
    }
    tots[cid] = 0;
    for (int j = 0; j < chunk; ++j) {
      tots[cid] += ns[start + j];
    }
  }

  barrier(nc);
  begin_roi();
  crs(val, vec, out, begin, ns, cid);
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

