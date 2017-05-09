#include <stdio.h>
#include "Constant.h"
#include "check.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

#define DTYPE int16_t

int main(int argc, char* argv[]) {
  DTYPE out[] = {0, 0,0,0,0,0,0}; 
  DTYPE in[]  = {1,200,300,4,5,600,700,8};

  //Version 1:
  begin_roi();
  SB_CONFIG(Constant_config,Constant_size);
  SB_DMA_READ(&in[0],8,8,2,P_Constant_in);
  SB_DMA_WRITE(P_Constant_out,8,8,2,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  //DTYPE expect[] = {10,200,300,10,10,600,700,10};
  DTYPE expect[] = {11,210,310,14,15,610,710,18};

  compare<DTYPE>(argv[0],out,expect,8);

  //printf("in: %lx out: %lx\n",*((uint64_t*)in),*((uint64_t*)out));
}
