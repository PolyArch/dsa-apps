#include "add.dfg.h"
#include "testing.h"

uint16_t answer[ASIZE];

int main(int argc, char* argv[]) {
  init();

  for(int i=0; i < ASIZE; ++i) {
    answer[i]=in[i]*2;
  }

  begin_roi();
  SB_CONFIG(add_config,add_size);
  SB_ADD_PORT(P_add_in2);
  SB_DMA_READ(&in[0],ABYTES,ABYTES,1,P_add_in1);
  SB_DMA_WRITE(P_add_out,8,8,AWORDS,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<DTYPE>(argv[0],out,answer,ASIZE);
}
