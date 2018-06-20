#include <stdio.h>
#include "add_ctrl.dfg.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>
#include <stdlib.h>

#define N 12

int main() {

  uint64_t x[N];
  uint64_t y[N];
  uint64_t a[N];
  for(uint64_t i=0; i<N; ++i) {
    x[i] = i;
    y[i] = i;
  }

  begin_roi();
  SB_CONFIG(add_ctrl_config,add_ctrl_size);
  
  SB_DMA_READ(&x[0], 8, 8, 1, P_add_ctrl_in1);
  // SB_DMA_READ(&x[0], 8, 8, N, P_add_ctrl_in1);
  SB_DMA_READ(&y[0], 8, 8, N, P_add_ctrl_in2);
  // SB_CONST(P_add_ctrl_p1, 2, N);
  SB_CONST(P_add_ctrl_p1, 0, N-1);
  SB_CONST(P_add_ctrl_p1, 2, 1);
  SB_DMA_WRITE(P_add_ctrl_out, 8, 8, N, &a[0]);
  SB_WAIT_ALL();
  end_roi();
}
