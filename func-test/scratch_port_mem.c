#include "testing.h"

int main(int argc, char* argv[]) {
  begin_roi();
  SB_CONFIG(none_config,none_size);
  SB_SCRATCH_READ(0, ABYTES, P_none_in);
  SB_DMA_WRITE(P_none_out,8,8,AWORDS,&out[0]);
  SB_WAIT_ALL();
  end_roi();
}
