#include <stdio.h>
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include "test.dfg.h"
#include <inttypes.h>
#define N 5


void sum(uint64_t x[],uint64_t y[], int s) { //,uint64_t z[],int s){
  uint64_t r[N];

  uint64_t ind1[N];
  uint64_t ind2[N];
  uint32_t ind3[N];

  for(uint64_t i=0; i<N; ++i){
    ind1[i] = i;
    ind2[i] = i;
  }

  for(uint32_t i=0; i<N; ++i){
    ind3[i] = i;
  }

  begin_roi();
  SB_CONFIG(test_config,test_size);

  // SB_DMA_READ(&ind1[0], 8, 8, N, P_IND_1);
  SB_DMA_READ(&ind3[0], 8, 8, N, P_IND_1);
  SB_DMA_READ(&ind2[0], 8, 8, N, P_IND_2);
  // SB_CONFIG_INDIRECT1(T64,T64,8,8);
  SB_CONFIG_INDIRECT1(T32,T64,8,8);
  SB_INDIRECT(P_IND_1, &x[0], N, P_test_A);
  SB_CONFIG_INDIRECT1(T64,T64,8,8);
  SB_INDIRECT(P_IND_2, &y[0], N, P_test_B);
  SB_DMA_WRITE(P_test_E,8,8,N,&r[0]);

  SB_WAIT_ALL();
  end_roi();

  sb_stats();
  for(int i=0;i<s/2;i++){
      printf("sum of the elements of a and b is: %ld\n",r[i]);
  }
}

int main()
{
    uint64_t a[2*N];
    uint64_t b[2*N];
    // uint64_t c[N];

    for (uint64_t i=0;i<(2*N-1);i+=2) {
      a[i] = (i*1+1);
      a[i+1] = (i*2+2);
      b[i] = (i*3+3);
      b[i+1] = (i*4+4);
    }   
    //printf("sum of the elements of a is: %d\n",sum(a,N));
    sum(a,b,2*N);
    return 0;
}
