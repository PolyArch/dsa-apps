#include "testing.h"
#include "add1.dfg.h"

static DTYPE mod[ASIZE];

int main(int argc, char* argv[]) {
  init();

  for(int i=0; i < ASIZE; ++i) {
    mod[i]=i%4+1;
  }

  SB_CONFIG(add1_config,add1_size);

  for(int i = 0; i < 3; ++i) {
    if(i==1) {
      begin_roi();
    }
    SB_CONST(P_add1_in,*((uint64_t*)in),AWORDS);
    SB_DMA_WRITE(P_add1_out,8,8,AWORDS,&out[0]);
    SB_WAIT_ALL();
  }
  end_roi();

  compare<DTYPE>(argv[0],out,mod,ASIZE);
  
}
