#include "testing.h"
#include <iostream>

int main(int argc, char* argv[]) {
  init();

  uint64_t ind_array[AWORDS*2];
  uint64_t data_array[AWORDS];
  uint64_t known[AWORDS*2],output[AWORDS*2];

  for(int i = 0; i<AWORDS*2; ++i) {
    known[i]=0;
    output[i]=0;
  }

  for(int i = 0; i<AWORDS; ++i) {
    ind_array[i]=i + i/4 + i/5 + i/11;
    //std::cout << i << "->" << ind_array[i] << "\n";
    known[ind_array[i]]=1;
  }

  for(int i = 0; i<AWORDS; ++i) {
    data_array[i]=1;
  }

  //for(int i = 0; i<AWORDS; ++i) {
  //  std::cout << i << "->" << ind_array[i] << "\n";
  //}

  std::cout << "outaddr:" << std::hex << output << "\n";

  begin_roi();
  SB_CONFIG(none_config,none_size);

  SB_DMA_READ(data_array,8,8,AWORDS,P_none_in);

  SB_DMA_READ(&ind_array[0],8,8,AWORDS,P_IND_1);
  SB_INDIRECT64_WR(P_IND_1,&output[0],AWORDS,P_none_out);

  SB_WAIT_ALL();
  end_roi();

  //for(int i = 0; i<AWORDS; ++i) {
  //  std::cout << i << "->" << ind_array[i] << "\n";
  //}


  compare<uint64_t>(argv[0],output,known,(int)AWORDS*2);
}