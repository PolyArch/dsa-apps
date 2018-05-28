#include <stdio.h>
#include "add_offset.dfg.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>
#include <stdlib.h>

#define N 12

int main() {

  uint64_t ind[N/2];
  for(uint64_t i=0; i<N/2; ++i) {
    ind[i] = i*2;
  }
  uint64_t x[N];
  for(uint64_t i=0; i<N; ++i) {
    x[i] = i;
  }

  begin_roi();
  SB_CONFIG(add_offset_config,add_offset_size);
  
  SB_CONST_SCR(0, 0, (64*2+2));
  SB_WAIT_SCR_WR();
  
  SB_DMA_READ(&ind[0], 8, 8, N/2, P_IND_1);
  SB_CONFIG_INDIRECT1(T64, T64, 1, 1);
  SB_INDIRECT(P_IND_1, &x[0], N/2, P_add_offset_A);
  
  SB_CONST(P_add_offset_const, 1, N/2);
  SB_ATOMIC_SCR_OP(P_add_offset_C, P_add_offset_D, 0, N, 0);
  SB_WAIT_SCR_WR();
  SB_WAIT_ALL();
  end_roi();
}
