#include "testing.h"

int main(int argc, char* argv[]) {
  init();

  begin_roi();
  SB_DMA_SCRATCH_LOAD(in,0,8,1,0);
  SB_WAIT_ALL();
  end_roi();
}
