#include <stdio.h>
#include "RShf64.dfg.h"
#include "check.h"
#include "../common/include/ss_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

#define DTYPE int16_t

int main(int argc, char* argv[]) {
  DTYPE out[] = {0, 0,0,0}; 
  DTYPE in[]  = {1,2,3,4};

  //Version 1:
  begin_roi();
  SS_CONFIG(RShf64_config,RShf64_size);
  SS_DMA_READ(&in[0],8,8,1,P_RShf64_in1);
  SS_CONST(P_RShf64_in2,16,1);
  SS_DMA_WRITE(P_RShf64_out,8,8,1,&out[0]);
  SS_WAIT_ALL();
  end_roi();

  DTYPE expect[] = {2,3,4,0};

  compare<DTYPE>(argv[0],out,expect,4);

  //printf("in: %lx out: %lx\n",*((uint64_t*)in),*((uint64_t*)out));
}
