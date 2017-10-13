#include <stdio.h>
#include "dot.h"
#include "dot_acc.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

#define ALEN 65536

uint16_t array1[ALEN]={};
uint16_t array2[ALEN]={};

uint64_t out = 0;

uint64_t accum[8]={0,0,0,0,0,0,0,0};
uint16_t accum16[4]={0,0,0,0};

int main(int argc, char* argv[]) {
  for(int i = 0; i < ALEN; ++i) {
    array1[i]=1;
    array2[i]=1;
  }

  uint16_t sum0=0; 
  uint64_t sum1=0, sum2=0, sum3=0;


  //begin_roi();
  //Version 0: (no softbrain)
  for(int i = 0; i < ALEN; ++i) {
    sum0 += array1[i] * array2[i];
  }
  //end_roi(); 



  //Version 1:                                                                                                                      
  begin_roi();                                                                                                                      
  SB_CONFIG(dot_config,dot_size);                                                                                                   
                                                                                                                                    
  SB_CONST(P_dot_carry,0,1);                                                                                                        
                                                                                                                                    
  SB_DMA_READ(array1, 8, 8, ALEN*sizeof(uint16_t)/8, P_dot_A);                                                                      
  SB_DMA_READ(array2, 8, 8, ALEN*sizeof(uint16_t)/8, P_dot_B);                                                                      
                                                                                                                                    
  SB_RECURRENCE(P_dot_R,P_dot_carry,ALEN/16-1);                                                                                     
  SB_DMA_WRITE(P_dot_R,8,8,1,&sum1);                                                                                                
                                                                                                                                    
  SB_WAIT_ALL();                                                                                                                    
  end_roi();          


  //Version 2:  (has scratchpad)
  if(ALEN < 2048) {
    int scr_addr=0;

    SB_CONST(P_dot_carry,0,1);

    SB_DMA_SCRATCH_LOAD(array1,8,8,ALEN*sizeof(uint16_t)/8,scr_addr);
    SB_WAIT_SCR_WR();

    SB_SCRATCH_READ(scr_addr, ALEN*sizeof(uint16_t), P_dot_A);
    SB_DMA_READ(array2, 4*sizeof(uint16_t), 4*sizeof(uint16_t), ALEN/4, P_dot_B);

    SB_RECURRENCE(P_dot_R,P_dot_carry,ALEN/16-1);
    SB_DMA_WRITE(P_dot_R,0,8,1,&sum2);

    SB_WAIT_ALL();
  }


  //Version 3: (uses accumulator)
  SB_CONFIG(dot_acc_config,dot_acc_size);

  SB_WAIT_ALL();

  SB_CONST(P_dot_acc_reset,0, ALEN/16-1);

  SB_DMA_READ(array1, 8, 8, ALEN*sizeof(uint16_t)/8, P_dot_acc_A);
  SB_DMA_READ(array2, 8, 8, ALEN*sizeof(uint16_t)/8, P_dot_acc_B);

  SB_CONST(P_dot_acc_reset,1, 1);

  SB_GARBAGE(P_dot_acc_R, ALEN/16-1);
  SB_DMA_WRITE(P_dot_acc_R,8,8,1,&accum16);

  SB_WAIT_ALL();

  sum3 = accum16[0] + accum16[1] + accum16[2] + accum16[3];

  printf("dot product (original): %d\n",sum0);
  printf("dot product (sb version 1): %d\n",sum1);
  printf("dot product (sb version 2): %d\n",sum2);
  printf("dot product (sb version 3): %d\n",sum3);

}
