#include <stdio.h>
#include "Acc16x4.h"
#include "check.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

#define DTYPE int16_t

int main(int argc, char* argv[]) {
  DTYPE out[] = {1,10,100,1000, 1,10,100,1000, 1,10,100,1000, 1,10,100,1000, 1,10,100,1000};
  DTYPE in[]  = {1,10,100,1000, 1,10,100,1000, 1,10,100,1000, 1,10,100,1000, 1,10,100,1000};

  //Version 1:
  begin_roi();
  SB_CONFIG(Acc16x4_config,Acc16x4_size);
  SB_DMA_READ(&in[0],8,8,5,P_Acc16x4_in);
  SB_CONST(P_Acc16x4_reset,0,2);
  SB_CONST(P_Acc16x4_reset,1,1);
  SB_CONST(P_Acc16x4_reset,0,2);
  SB_DMA_WRITE(P_Acc16x4_out,8,8,5,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  DTYPE expect[] = {1,10,100,1000, 2,20,200,2000, 3,30,300,3000, 1,10,100,1000, 2,20,200,2000};

  compare<DTYPE>(argv[0],out,expect,4*5);

  //printf("in: %lx out: %lx\n",*((uint64_t*)in),*((uint64_t*)out));
}