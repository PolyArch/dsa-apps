#include "testing.h"
#include <iostream>

#define NARR 32
#define NIND 8

uint32_t ind_array[NIND];
//output data type is 16 bits

uint16_t known[NARR]={0};
uint16_t output[NARR]={0};

int main(int argc, char* argv[]) {
  init();

  ind_array[0]=1;
  ind_array[1]=1;
  ind_array[2]=2;
  ind_array[3]=3;
  ind_array[4]=5;
  ind_array[5]=8;
  ind_array[6]=13;
  ind_array[7]=21;

  for(int i = 0; i<NIND; ++i) {
    known[ind_array[i]]=3; 
  }

  SB_CONFIG(none_config,none_size);

  SB_CONST(P_none_in,0x0003000300030003,NIND/4);

  begin_roi();
  SB_DMA_READ(&ind_array[0],8,8,NIND/2,P_IND_1);

  //itype, dtype, mult, offset
  SB_CONFIG_INDIRECT(T32,T16,sizeof(uint16_t));  //indexing into 16-bit data
  SB_INDIRECT_WR_SCR(P_IND_1,0/*begining of scratch*/,NIND,P_none_out);

  SB_WAIT_SCR_WR();

  SB_SCRATCH_DMA_STORE(0,8,8,2*NARR/8,&output[0]);
  SB_WAIT_ALL();

  end_roi();

  compare<uint16_t>(argv[0],output,known,NARR);
}
