#include "testing.h"
#include "acc_up.dfg.h"

void test() {
  init();
  SS_CONFIG(acc_up_config,acc_up_size);

  SS_CONST(P_acc_up_reset,0,ASIZE/4-1);
  SS_DMA_READ(&in[0],ABYTES,ABYTES,1,P_acc_up_in);
  SS_CONST(P_acc_up_reset,1,1);
  SS_DMA_WRITE(P_acc_up_out,16,16,1,&out[0]);
  SS_WAIT_ALL();
}

int main(int argc, char* argv[]) {
  
  static uint32_t out32[4]; 

  for(int i = 0; i < ASIZE; i+=4) {
    out32[0] += i+0;
    out32[1] += i+1;
    out32[2] += i+2;
    out32[3] += i+3;
  }


  test();
  compare<uint32_t>(argv[0], out32, (uint32_t*)out, 4);

  return 0;
}
