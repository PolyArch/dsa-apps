#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/multicore.h"
#define FAKE
#include "../Specs/i16-128x128x4.h"

#ifndef U
#define U 1
#endif

TYPE src[N + col_size * 2];

TYPE Ix[N + col_size * 2];
TYPE Iy[N + col_size * 2];
TYPE Ixy[N + col_size * 2];

TYPE derH[N + col_size * 2];
TYPE derV[N + col_size * 2];

void derivative(int cid) {


  // The base workload each core should do.
  int64_t chunk = row_size / NUM_CORES;
  // The additional residues that some cores should do.
  int64_t residue = row_size % NUM_CORES;
  // The starting address of this core (cid).
  int64_t start = cid * chunk + mc_min(residue, cid);

#pragma ss config
  {
    int64_t r1;
#pragma ss stream
    for (int64_t ri = 0; ri < chunk; ri++) {
#pragma ss dfg dedicated unroll(4)
      for (int64_t c = 0; c < col_size; c++) {
        r1 = (ri + start) * col_size;
        TYPE a1 = src[r1 - col_size + c];
        TYPE a2 = src[r1 + c];
        TYPE a3 = src[r1 + col_size + c];
        derV[r1 - col_size + c] = a1 + a2 /*+ a2*/ + a3;
      }
    }
  }

#pragma ss config
  {
    for (int64_t ri = 0; ri < chunk; ri++) {
      int64_t r = ri + start;
      #pragma ss stream nonblock
      #pragma ss dfg dedicated unroll(1)
      for (int64_t c = 1; c < col_size - 1; c += 2) {
        int64_t rowOffset = r * col_size;
        TYPE a1 = src[rowOffset + c - 1];
        TYPE a2 = src[rowOffset + c];
        TYPE a3 = src[rowOffset + c + 1];
        TYPE a4 = src[rowOffset + c + 2];
        derH[rowOffset + c - 1] = a1 + a2 /*+ a2*/ + a3;
        derH[rowOffset + c] = a2 + a3 /*+ a2*/ + a4;
      }
    }
  }
  
#pragma ss config
  {
 #pragma ss stream
    for (int64_t rr = 0; rr < chunk; rr++) {
 #pragma ss dfg dedicated unroll(2)
      for (int64_t c = 0; c < col_size - 2; c++) {
        int64_t rowOffset = (rr + start) * col_size;
        Ix[rowOffset + c] = derH[rowOffset + c] - derH[rowOffset + 2 * col_size + c];
        Iy[rowOffset + c] = derV[rowOffset + c + 2] - derV[rowOffset + c];
        Ixy[rowOffset + c] = Ix[rowOffset + c] * Iy[rowOffset + c];
      }
    }
  }
}

void thread_entry(int cid, int nc) {

  barrier(nc);
  begin_roi();
  derivative(cid);
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

