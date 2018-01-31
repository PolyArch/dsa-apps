#include "testing.h"
#include "mul_by_2.h"

static DTYPE answer[ASIZE]; 

int main(int argc, char* argv[]) {
  init();

  for(int i = 0; i<ASIZE; ++i) {
    answer[i]=i*2;
  }

  begin_roi();
  SB_CONFIG(mul_by_2_config,mul_by_2_size);
  SB_DMA_READ(&in[0],ABYTES,ABYTES,1,P_mul_by_2_in);
  SB_DMA_WRITE(P_mul_by_2_out,8,8,AWORDS,&out[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<DTYPE>(argv[0],out,answer,ASIZE);
}
