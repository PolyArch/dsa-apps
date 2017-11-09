#include "testing.h"

static DTYPE mod[ASIZE];

int main(int argc, char* argv[]) {
  init();

  for(int i=0; i < ASIZE; ++i) {
    mod[i]=i%4;
  }

  begin_roi();
  SB_CONFIG(none_config,none_size);
  SB_CONST(P_none_in,*((uint64_t*)in),AWORDS);
  SB_DMA_WRITE(P_none_out,8,8,AWORDS,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<DTYPE>(argv[0],out,mod,ASIZE);
  
}
