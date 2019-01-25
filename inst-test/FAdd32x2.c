#include <stdio.h>
#include "FAdd32x2.dfg.h"
#include "check.h"
#include "../common/include/ss_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

#define DTYPE float

int main(int argc, char* argv[]) {
  DTYPE out[] = {0, 0,0,0}; 
  DTYPE in1[]  = {1,200,300,4};
  DTYPE in2[]  = {100,2,3,400};

  //Version 1:
  begin_roi();
  SS_CONFIG(FAdd32x2_config,FAdd32x2_size);
  SS_DMA_READ(&in1[0],8,8,2,P_FAdd32x2_in1);
  SS_DMA_READ(&in2[0],8,8,2,P_FAdd32x2_in2);
  SS_DMA_WRITE(P_FAdd32x2_out,8,8,2,&out[0]);
  SS_WAIT_ALL();
  end_roi();

  DTYPE expect[] = {101,202,303,404};

  compare<DTYPE>(argv[0],out,expect,4);

  //printf("in: %lx out: %lx\n",*((uint64_t*)in),*((uint64_t*)out));
}
