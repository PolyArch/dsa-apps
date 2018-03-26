#include "testing.h"

static uint64_t input[AWORDS];
static uint64_t output[AWORDS];
static uint64_t answer[AWORDS];

int main(int argc, char* argv[]) {
  for(int i=0; i < AWORDS; ++i) {
    input[i]=i;
    output[i]=0;
    answer[AWORDS-i-1]=i;
  }

  begin_roi();
  SB_CONFIG(none_config,none_size);
  SB_DMA_SCRATCH_LOAD(input,ABYTES,ABYTES,1,0);
  SB_WAIT_SCR_WR();

  //SB_SCRATCH_READ(ABYTES, ABYTES, P_none_in);
  SB_SCR_PORT_STREAM_STRETCH(ABYTES, -8,-8,0,AWORDS, P_none_in)

  SB_DMA_WRITE(P_none_out,ABYTES,ABYTES,1,&output[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<uint64_t>(argv[0],output,answer,AWORDS);

  sb_stats();
}
