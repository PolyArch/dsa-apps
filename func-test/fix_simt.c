#include "testing.h"

int main(int argc, char* argv[]) {
  init();

  SB_CONTEXT_OFFSET(0x0003,ABYTES/2);
  SB_CONFIG(none_config,none_size);

  begin_roi();
  SB_DMA_READ(&in[0],ABYTES/2,ABYTES/2,1,P_none_in);
  SB_DMA_WRITE(P_none_out,8,8,AWORDS/2,&out[0]);
  SB_WAIT_ALL();
  end_roi();
 
  compare<DTYPE>(argv[0],out, in,ASIZE);
}
