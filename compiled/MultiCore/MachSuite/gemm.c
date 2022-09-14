/*
Implementation based on algorithm described in:
The cache performance and optimizations of blocked algorithms
M. D. Lam, E. E. Rothberg, and M. E. Wolf
ASPLOS 1991
*/

#include <stdint.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#include "../Specs/gemm.h"

__attribute__((section(".noinit")))
TYPE a[N], b[N], c[N];

#ifndef U
#define U 4
#endif

void gemm(TYPE a[N], TYPE b[N], TYPE prod[N], int64_t cid) {

#define PART NUM_CORES
#define CHUNK (col_size / T / PART)
#define T 2
#define B_CHUNK ((col_size / T) * CHUNK * T * T)


  // The base workload each core should do.
  int64_t chunk = (row_size / T) / NUM_CORES;
  // The additional residues that some cores should do.
  int64_t residue = (row_size / T) % NUM_CORES;
  // The starting address of this core (cid).
  int64_t start = cid * chunk + mc_min(residue, cid);
  // Distribute the addtional residue.
  chunk += cid < residue;


  #pragma ss config
  {
    TYPE spad[B_CHUNK];
    #pragma ss stream
    #pragma ss dfg unroll(4)
    for (int64_t i = 0; i < col_size; ++i) {
      spad[i] = b[i + B_CHUNK * cid];
    }
    // for (int64_t jo = 0; jo < PART; ++jo) {
    {
      int64_t jo = cid;
      for (int64_t ii = 0; ii < chunk; ++ii) {
        int64_t i = ii + start;
        #pragma ss stream nonblock
        for (int64_t k = 0; k < col_size / T; ++k) {
          #pragma ss dfg
          for (int64_t ji = 0; ji < CHUNK; ++ji) {
            int64_t j = jo * CHUNK + ji;
            #define decl_load(array, idx) TYPE v##array##idx = v##array[idx]
            TYPE *va = &a[i * ((col_size / T) * T * T) + k * T * T]; // [i, k]
            decl_load(a, 0); decl_load(a, 1); decl_load(a, 2); decl_load(a, 3);
            TYPE *vb = &spad[k * CHUNK * T * T + ji * T * T]; // [k, j]
            decl_load(b, 0); decl_load(b, 1); decl_load(b, 2); decl_load(b, 3);
            TYPE *vc = &c[i * ((col_size / T) * T * T) + j * T * T]; // [i, j]
            decl_load(c, 0); decl_load(c, 1); decl_load(c, 2); decl_load(c, 3);
            vc[0] += va0 * vb0 + va1 * vb2;
            vc[1] += va0 * vb1 + va1 * vb3;
            vc[2] += va2 * vb0 + va3 * vb2;
            vc[3] += va2 * vb1 + va3 * vb3;
          }
        }
      }
    }
  }
}

void thread_entry(int cid, int nc){
  barrier(nc);
  begin_roi();
  gemm(a, b, c, cid);
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

