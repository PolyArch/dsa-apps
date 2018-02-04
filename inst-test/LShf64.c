#include <stdio.h>
#include "LShf64.dfg.h"
#include "check.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

#define DTYPE int16_t

int main(int argc, char* argv[]) {
  DTYPE out[] = {0, 0,0,0}; 
  DTYPE in[]  = {1,2,3,4};

  //Version 1:
  begin_roi();
  SB_CONFIG(LShf64_config,LShf64_size);
  SB_DMA_READ(&in[0],8,8,1,P_LShf64_in1);
  SB_CONST(P_LShf64_in2,16,1);
  SB_DMA_WRITE(P_LShf64_out,8,8,1,&out[0]);
  SB_WAIT_ALL();

  DTYPE expect[] = {0,1,2,3};

  compare<DTYPE>(argv[0],out,expect,4);

  DTYPE in2[]  = {0xff,2,3,0xff};

  SB_DMA_READ(&in2[0],8,8,1,P_LShf64_in1);
  SB_CONST(P_LShf64_in2,64,1);
  SB_DMA_WRITE(P_LShf64_out,8,8,1,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  DTYPE expect2[] = {0,0,0,0};

  compare<DTYPE>(argv[0],out,expect2,4);


  //printf("in: %lx out: %lx\n",*((uint64_t*)in),*((uint64_t*)out));
}
