#include "testing.h"
#include "add1.h"

static DTYPE add1[ASIZE]; 
static DTYPE answer_add1[ASIZE]; 

static DTYPE out2[ASIZE]; 


int main(int argc, char* argv[]) {
  init();

  for(int i=0; i < ASIZE; ++i) {
    answer_add1[i]=i+1;
  }

  SB_CONTEXT(0x0001);
  SB_CONFIG(none_config,none_size);
  SB_CONTEXT(0x0002);
  SB_CONFIG(add1_config,add1_size);

  begin_roi();

  SB_CONTEXT(0x0001);
  SB_DMA_READ(&in[0],ABYTES,ABYTES,1,P_none_in);

  SB_XFER_RIGHT(P_none_out,P_add1_in,AWORDS);

  SB_CONTEXT(0x0002);
  SB_DMA_WRITE(P_add1_out,8,8,AWORDS,&out2[0]);

  SB_CONTEXT(0x0003);
  SB_WAIT_ALL();

  end_roi();

  compare<DTYPE>(argv[0],out2,answer_add1,ASIZE);
}