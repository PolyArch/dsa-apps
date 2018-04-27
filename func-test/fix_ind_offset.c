#include "testing.h"
#include <iostream>

#define LEN 128

struct item_array {
  uint64_t item1;
  uint64_t item2;
  uint64_t item3;
  uint64_t item4;
} array[LEN];

int main(int argc, char* argv[]) {
  init();

  uint64_t ind_array[LEN];
  uint64_t known[LEN*2];
  uint64_t output[LEN*2];


  for(int i = 0; i<LEN; ++i) {
    array[i].item1 = i+1;
    array[i].item2 = i+2;
    array[i].item3 = i+3;
    array[i].item4 = i+4;

    ind_array[i]=LEN-i-1;

    //assume offsets 1 and 4
    known[2*i+0]=(LEN-i-1)+1;
    known[2*i+1]=(LEN-i-1)+4;
    output[2*i+0]=-1;
    output[2*i+1]=-1;
  }

  SB_CONFIG(none_config,none_size);

  begin_roi();
  SB_DMA_READ(&ind_array[0],8,8,LEN,P_IND_1);

  SB_CONFIG_INDIRECT1(T64,T64,sizeof(item_array), 24); //itype, dtype, mult, offset
  SB_INDIRECT(P_IND_1,&array[0],LEN,P_none_in);

  SB_DMA_WRITE(P_none_out,8,8,LEN*2,&output[0]);
  SB_WAIT_ALL();
  end_roi();

  compare<uint64_t>(argv[0],output,known,(int)LEN);
}
