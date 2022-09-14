#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

#include "dsaintrin.h"
#include "common/timing.h"
#include "./add.dfg.h"

#define N 128

int64_t a[N], b[N], c[N], ref[N];

int main() {
  for (int i = 0; i < N; ++i) {
    a[i] = rand();
    b[i] = rand();
    ref[i] = a[i] + b[i];
  }
  
  begin_roi();
  SS_CONFIG(add_config, add_size);
  INSTANTIATE_1D_STREAM(/*start-addr=*/ a,
                        /*stride=*/ (uint64_t) 1,
                        /*n1d=*/ (uint64_t) N,
                        /*port=*/ P_compute_A,
                        /*padding=*/ DP_NoPadding,
                        /*action=*/ DSA_Access,
                        /*operation=*/ DMO_Read,
                        /*memory-ty=*/ DMT_DMA,
                        /*dtype=*/ sizeof(int64_t),
                        /*ctype=*/ 0);
  INSTANTIATE_1D_STREAM(/*start-addr=*/ b,
                        /*stride=*/ (uint64_t) 1,
                        /*n1d=*/ (uint64_t) N,
                        /*port=*/ P_compute_A,
                        /*padding=*/ DP_NoPadding,
                        /*action=*/ DSA_Access,
                        /*operation=*/ DMO_Read,
                        /*memory-ty=*/ DMT_DMA,
                        /*dtype=*/ sizeof(int64_t),
                        /*ctype=*/ 0);
  INSTANTIATE_1D_STREAM(/*start-addr=*/ c,
                        /*stride=*/ (uint64_t) 1,
                        /*n1d=*/ (uint64_t) N,
                        /*port=*/ P_compute_A,
                        /*padding=*/ DP_NoPadding,
                        /*action=*/ DSA_Access,
                        /*operation=*/ DMO_Write,
                        /*memory-ty=*/ DMT_DMA,
                        /*dtype=*/ sizeof(int64_t),
                        /*ctype=*/ 0);
  SS_WAIT_ALL();
  end_roi();
  ss_stats();


  for (int i = 0; i < N; ++i) {
    assert(ref[i] == c[i]);
  }

  return 0;
}
