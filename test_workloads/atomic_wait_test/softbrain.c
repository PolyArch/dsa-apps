#include <stdio.h>
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include "test.dfg.h"
#include <inttypes.h>
#define N 5
#define m 3


void sum(uint64_t x[],uint64_t y[],int s){
  uint64_t z[s];

  begin_roi();
  SB_CONST_SCR(0,5,5);
  SB_WAIT_SCR_WR();
  SB_WAIT_SCR_RD();

  SB_CONFIG(test_config,test_size);

  SB_DMA_READ(&x[0],8,8,s,P_test_A);
  SB_DMA_READ(&y[0],8,8,s,P_test_B);
  SB_DMA_WRITE(P_test_C,8,8,s+2,&z[0]);
  // SB_DMA_WRITE(P_test_C,8,8,s,&z[0]);

  SB_CONST(P_test_addr, 0, m);
  SB_CONST(P_test_val, 1, m);
  SB_CONFIG_ATOMIC_SCR_OP(T64, T64, T64);
  SB_ATOMIC_SCR_OP(P_test_D, P_test_E, 0, m, 0);

  SB_WAIT_SCR_ATOMIC();
  // so that there is no value left at the ports

  SB_RESET();
  
  SB_WAIT_ALL();
  end_roi();

  for(int i=0;i<s;i++){
      printf("sum of the elements of a and b is: %ld\n",z[i]);
  }

  sb_stats();
}

int main()
{
    uint64_t a[N];
    uint64_t b[N];
    for (int i=0;i<N;i++) {
      a[i]=i*5;
      b[i]=i*2;
    }   
    //printf("sum of the elements of a is: %d\n",sum(a,N));
    sum(a,b,N);
    return 0;
}


