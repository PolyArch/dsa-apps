#include "testing.h"

DTYPE str2[ASIZE];

int main(int argc, char* argv[]) {
  init();

  for(int i = 0; i < A2WORDS/sizeof(DTYPE)*8; i+=1) {
    str2[i]=(i/4)*8+i%4;
  }

  begin_roi();
  SB_CONFIG(none_config,none_size);
  SB_DMA_READ(&in[0],16,8,A2WORDS,P_none_in);
  SB_DMA_WRITE(P_none_out,8,8,A2WORDS,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<DTYPE>(argv[0],out,str2,A2WORDS/sizeof(DTYPE)*8);
}
