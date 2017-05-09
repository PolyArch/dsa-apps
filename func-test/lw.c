#include "testing.h"

int main(int argc, char* argv[]) {
  init();

  begin_roi();
  SB_CONFIG(none_config,none_size);
  SB_DMA_READ(&in[0],ABYTES,ABYTES,1,P_none_in);
  SB_DMA_WRITE(P_none_out,8,8,AWORDS,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<DTYPE>(argv[0],out,in,ASIZE);
}
