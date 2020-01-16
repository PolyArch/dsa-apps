#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "none.dfg.h"

#define NUM_THREADS 2

#define N 8
#define SRC 0
#define LIM 8
#define DST_LOC 5

uint64_t a[2*N];
uint64_t b[2*N];

// Barrier variable
pthread_barrier_t barr;
pthread_barrier_t barr2;

uint64_t mask=0;

void remPort(long tid) {
   begin_roi();
   int scr_offset=0; // for 0th core
   if(tid==1){
     // ind_port, addr_offset, num_elem, input_port
     SS_DMA_READ(&a[0], 8, 8, N, P_IND_1);
     SS_CONFIG_INDIRECT(T64, T64, 8);
     SS_INDIRECT_SCR(P_IND_1, scr_offset, N, P_none_in);
     SS_DMA_WRITE(P_none_out, 8, 8, N, &b[0]);
   }
   SS_WAIT_ALL();
   SS_GLOBAL_WAIT(NUM_THREADS);

   end_roi();
   sb_stats();
}

void *entry_point(void *threadid) {
  
   long tid;
   tid = (long)threadid;

   // int rc2 = pthread_barrier_wait(&barr2);

   SS_CONFIG(none_config,none_size);
   SS_GLOBAL_WAIT(NUM_THREADS);
 
   // Synchronization point
   int rc = pthread_barrier_wait(&barr);
   if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
   {
     printf("Could not wait on barrier\n");
     // exit(-1);
   }

  // should be configured before starting any core
   remPort(tid);
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
  for(uint64_t i=0; i<N; i++){
    a[i] = i;
    b[i] = i;
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
