#include "testing.h"
#include "add1.h"

static DTYPE add1[ASIZE]; 
static DTYPE answer_add1[ASIZE]; 

static DTYPE out2[ASIZE]; 


int main(int argc, char* argv[]) {
  init();

  if(AWORDS&1) {
    return 0; //not valid for this case
  }

  for(int i=0; i < AWORDS; ++i) {
    for(int j=0; j < 4; ++j) {
      answer_add1[i*4+j]=(i/2)*4+j+1;
    }
  }

  SB_CONTEXT(0x0001);
  SB_CONFIG(add1_config,add1_size);

  SB_CONTEXT(0x0002);
  SB_CONFIG(none_config,none_size);

  begin_roi();



  SB_CONTEXT(0x0002);
  SB_DMA_READ(&in[0],ABYTES/2,ABYTES/2,1,P_none_in);

  SB_REPEAT_PORT(2);
  SB_XFER_LEFT(P_none_out,P_add1_in,AWORDS/2);


  SB_CONTEXT(0x0001);
  SB_DMA_WRITE(P_add1_out,8,8,AWORDS,&out2[0]);

  SB_CONTEXT(0x0003);
  SB_WAIT_ALL();

  end_roi();

  compare<DTYPE>(argv[0],out2,answer_add1,ASIZE);
}
