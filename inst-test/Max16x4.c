#include <stdio.h>
#include "Max16x4.dfg.h"
#include "check.h"
#include "../common/include/ss_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

#define DTYPE int16_t

int main(int argc, char* argv[]) {
  DTYPE out[] = {0, 0,0,0}; 
  DTYPE in1[]  = {1,200,300,4};
  DTYPE in2[]  = {100,2,3,400};

  //Version 1:
  begin_roi();
  SS_CONFIG(Max16x4_config,Max16x4_size);
  SS_DMA_READ(&in1[0],8,8,1,P_Max16x4_in1);
  SS_DMA_READ(&in2[0],8,8,1,P_Max16x4_in2);
  SS_DMA_WRITE(P_Max16x4_out,8,8,1,&out[0]);
  SS_WAIT_ALL();
  end_roi();

  DTYPE expect[] = {100,200,300,400};

  compare<DTYPE>(argv[0],out,expect,4);

  //printf("in: %lx out: %lx\n",*((uint64_t*)in),*((uint64_t*)out));
}
