#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Specs/fir.h"

#define rec_block 32

TYPE a[N], b[M], c[N - M + 1 + ((rec_block - (N - M + 1) % rec_block) % rec_block)];

void fir(TYPE a[N], TYPE b[M], TYPE c[N - M + 1], int64_t cid) {
  #pragma ss config
  {
    int64_t chunk = (N - M + 1) / rec_block / NUM_CORES;
    int64_t start_with = 0;
    int64_t residue = (N - M + 1) / rec_block % NUM_CORES;
    if (cid < residue) {
      chunk++;
      start_with = cid * chunk;
    } else {
      start_with = cid * chunk + residue;
    }
    TYPE spad[rec_block * ((N - M) / NUM_CORES + 1) - M + 1];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int64_t i = 0; i < chunk * rec_block - M + 1; ++i) {
      spad[i] = a[start_with + i];
    }
    for (int64_t io = 0; io < chunk; ++io) {
      #pragma ss stream nonblock
      for (int64_t j = 0; j < M; ++j) {
        #pragma ss dfg dedicated unroll(4)
        for (int64_t ii = 0; ii < rec_block; ++ii) {
          int64_t i = (io * rec_block + ii);
          c[i] += spad[i + j] * b[j];
        }
      }
    }
  }
}

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  fir(a, b, c, cid);
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

