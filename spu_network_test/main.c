#include <stdio.h>
// #include <omp.h>
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "none.dfg.h"

#define N 5

int main(){
  uint64_t a[N];
  uint64_t b[N];
  
  for(uint64_t i=0; i<N; i++){
    a[i] = i;
    b[i] = i;
  }

  SB_CONFIG(none_config,none_size);

  SB_DMA_READ_SIMP(&a[0], N, P_none_A);
  // SB_REM_PORT(output_port, num_elem, mask, remote_port)
  // SB_REM_PORT(P_none_B, N, 1, P_none_A);
  SB_DMA_WRITE_SIMP(P_none_B, N, &b[0]);
  SB_WAIT_ALL();
  return 0;
};
