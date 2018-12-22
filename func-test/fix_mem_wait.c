#include "testing.h"

int main(int argc, char* argv[]) {
  init();

  begin_roi();
  SS_DMA_SCRATCH_LOAD(in,0,8,1,0);
  SS_WAIT_ALL();
  end_roi();
}
