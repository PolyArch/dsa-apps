#include "testing.h"

int main(int argc, char* argv[]) {
  init();

  begin_roi();
  SB_CONFIG(none_config,none_size);
  SB_DMA_SCRATCH_LOAD(in,ABYTES,ABYTES,1,0);
  SB_WAIT_SCR_WR();
  //SB_WAIT_ALL();
  SB_SCRATCH_DMA_STORE(0,8,8,AWORDS,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<DTYPE>(argv[0],out,in,ASIZE);
}
