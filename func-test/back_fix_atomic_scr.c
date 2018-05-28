#include <stdio.h>
#include "add_offset.dfg.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>
#include <stdlib.h>

#define N 12

int main() {

  uint64_t x[N];
  // uint64_t a[N]; uint64_t b[N];
  for(uint64_t i=0; i<N; ++i) {
    x[i] = i;
  }

  begin_roi();
  SB_CONFIG(add_offset_config,add_offset_size);
  
  SB_DMA_READ(&x[0], 8, 8, N, P_add_offset_A);
  
  SB_CONST(P_add_offset_const, 1, N/2);
  // SB_DMA_WRITE(P_add_offset_C, 8, 8, N, &a[0]);
  // SB_DMA_WRITE(P_add_offset_D, 8, 8, N, &b[0]);
  SB_ATOMIC_SCR_OP(P_add_offset_C, P_add_offset_D, 0, N, 0);
  SB_WAIT_SCR_WR();
  SB_WAIT_ALL();
  end_roi();
}
