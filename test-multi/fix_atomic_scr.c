#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "none_64.dfg.h"

#define NUM_THREADS 4

#define N 8
#define SRC 0
#define LIM 6
#define DST_LOC 5

uint64_t a[2*N];
uint64_t b[2*N];
uint64_t c[2*N];

// Barrier variable
pthread_barrier_t barr;

uint64_t mask=0;
int scr_offset=(32768-16)/8;

void remPort(long tid) {
   if(tid==SRC){
     SS_DMA_READ(&a[tid*N], 8, 8, N*2, P_none_64_A); // addr
     SS_CONST(P_none_64_C, 2, N*2);  // val
     
     SS_CONFIG_ATOMIC_SCR_OP(T64, T64, T64, 2, 2, 0);
     SS_ATOMIC_SCR_OP(P_none_64_B, P_none_64_D, scr_offset, N, 0); // total updates = val_num*N

     // SS_CONFIG_ATOMIC_SCR_OP(T64, T64, T64, 2, 2, 0); // <addr,val> = <0,0>, <16,0>, <8,1>, <24,1>, <32,2>, <40,2>
     // <val,vec<dest>> => <(0,16),0> <(8,24), 1>.... => coalesce requests in
     // the address dimension, broadcast along the value dimension (send vec<dest>, num_bytes)
 
   } else if (tid < LIM) {
     // Why would it check atomic request banks...
   }
   SS_WAIT_ALL();
   SS_GLOBAL_WAIT(NUM_THREADS); // may be when it reached here, spu sent was 0 (basically i want to make sure if all cores are done, oh that's ok)

}

void remPort2(long tid) {
  uint8_t x[N];
  for(int i=0; i<N; ++i) x[i] = 2;
  x[0]=1; x[1]=3;
  if(tid==SRC){
    SS_DMA_READ(&a[tid*N], 8, 8, N*2, P_none_64_A); // addr
    SS_CONST(P_none_64_C, 2, N*2);  // val
   
    // SS_CONST(P_IND_1, 2, N); // N times
    SS_DMA_READ(&x[0], 1, 1, N, P_IND_1); // addr
    SS_CONFIG_ATOMIC_SCR_OP(T64, T64, T64, 2, P_IND_1, 1); // TODO: check at port, (also variable times)
    SS_ATOMIC_SCR_OP(P_none_64_B, P_none_64_D, scr_offset, N, 0); // total updates = val_num*N

    // SS_CONFIG_ATOMIC_SCR_OP(T64, T64, T64, 2, 2, 0); // <addr,val> = <0,0>, <16,0>, <8,1>, <24,1>, <32,2>, <40,2>
    // <val,vec<dest>> => <(0,16),0> <(8,24), 1>.... => coalesce requests in
    // the address dimension, broadcast along the value dimension (send vec<dest>, num_bytes)
 
  } else if (tid < LIM) {
    // Why would it check atomic request banks...
  }
  SS_WAIT_ALL();
  SS_GLOBAL_WAIT(NUM_THREADS); // may be when it reached here, spu sent was 0 (basically i want to make sure if all cores are done, oh that's ok)

}

// test different data type of bytes and atomic scr op
void remPort3(long tid) {
  uint8_t x[N];
  for(int i=0; i<N; ++i) x[i] = 2;
  x[0]=1; x[1]=3;
  if(tid==SRC){
    // SS_DMA_READ(&a[tid*N], 1, 1, N*2*8, P_IND_2); // addr
    // SS_DMA_READ_SIMP(&a[tid*N], N*2*8, P_IND_2); // addr
    SS_CONST(P_IND_3, 2, N*2);  // val
    SS_CONST(P_IND_2, 2, N*2);  // val
   
    SS_CONST(P_IND_1, 2, N); // N times
    // SS_DMA_READ_SIMP(&x[0], N, P_IND_1); // addr
    SS_CONFIG_ATOMIC_SCR_OP(T08, T08, T08, 2, P_IND_1, 1); // TODO: check at port, (also variable times)
    SS_ATOMIC_SCR_OP(P_IND_2, P_IND_3, scr_offset, N, 0); // total updates = val_num*N

    // SS_CONFIG_ATOMIC_SCR_OP(T64, T64, T64, 2, 2, 0); // <addr,val> = <0,0>, <16,0>, <8,1>, <24,1>, <32,2>, <40,2>
    // <val,vec<dest>> => <(0,16),0> <(8,24), 1>.... => coalesce requests in
    // the address dimension, broadcast along the value dimension (send vec<dest>, num_bytes)
 
  } else if (tid < LIM) {
    // Why would it check atomic request banks...
  }
  SS_WAIT_ALL();
  SS_GLOBAL_WAIT(NUM_THREADS); // may be when it reached here, spu sent was 0 (basically i want to make sure if all cores are done, oh that's ok)

}

void *entry_point(void *threadid) {
  
   long tid;
   tid = (long)threadid;

   SS_CONFIG(none_64_config,none_64_size);
   SS_GLOBAL_WAIT(NUM_THREADS);

 
   // Synchronization point
   int rc = pthread_barrier_wait(&barr);
   if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
   {
     printf("Could not wait on barrier\n");
     // exit(-1);
   }



  // should be configured before starting any core
   begin_roi();
   /*
   remPort(tid);
   printf("Safely completed test1\n");
   remPort2(tid);
   printf("Safely completed test2\n");
   */
   remPort3(tid);
   end_roi();
   sb_stats();
   // pthread_exit(NULL);
   return NULL;
}


int main(){

  // addDest(mask, DST_LOC);
  // for(int i=0; i<NUM_THREADS; ++i) {
  for(int i=0; i<LIM; ++i) {
    if(i!=SRC) addDest(mask,i);
  }


  // data generation
  for(uint64_t i=0; i<2*N; i++){
    a[i] = 2*i;
    b[i] = i;
    c[i] = i;
  }
  
  // assert(NUM_THREADS<C);
  
  // Barrier initialization
  if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }

  pthread_t threads[NUM_THREADS];
  int rc;
  long t;
  
  for(t=0;t<NUM_THREADS;t++){
    printf("In main: creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
	  return 0;
    }
  }
   
  for(int i = 0; i < NUM_THREADS; ++i) {
    if(pthread_join(threads[i], NULL)) {
  	printf("Could not join thread %d\n", i);
      return -1;
    }
  }

  return 0;
};
