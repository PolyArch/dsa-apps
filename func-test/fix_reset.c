#include "testing.h"
#include "add1_vec.dfg.h"

uint64_t input[4];
uint64_t output[4];
uint64_t answer[4];


int main(int argc, char* argv[]) {
  init();
  input[0]=1;
  input[1]=2;
  input[2]=3;
  input[3]=4;

  answer[0]=2;
  answer[1]=3;
  answer[2]=4;
  answer[3]=5;


  begin_roi();
  SB_CONFIG(add1_vec_config,add1_vec_size);
  SB_DMA_READ(&input[0],8,8,  4, P_add1_vec_in);
  SB_DMA_WRITE(P_add1_vec_out,8,8, 4,&output[0]);

  //These are all garbage commands
  SB_DMA_SCRATCH_LOAD(in,ABYTES,ABYTES,1,0);
  SB_DMA_READ(&input[0],8,8, 1, P_add1_vec_in);
  SB_SCRATCH_READ(0, ABYTES, P_none_in);
  SB_DMA_READ(&input[0],8,8, 4,P_add1_vec_in);

  SB_WAIT_MEM_WR();
  SB_RESET();
  SB_WAIT_ALL();
  end_roi();

  compare<uint64_t>(argv[0],answer,output,4);
}
