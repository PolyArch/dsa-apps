#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "none_8.dfg.h"

#define NUM_THREADS 2

#define N 8

uint64_t a[2*N];
uint64_t b[2*N];

// Barrier variable
pthread_barrier_t barr;


void remPort(long tid) {

  uint64_t mask=0;
  for(int i=1; i<NUM_THREADS; ++i) addDest(mask,i);

   begin_roi();
   SS_CONFIG(none_8_config,none_8_size);
   SS_STRIDE(1,1);
   if(tid==0){
     SS_DMA_READ_SIMP(&a[tid*N], N, P_none_8_A);
     SS_REM_PORT(P_none_8_B, N, mask, P_none_8_A);
     SS_DMA_READ_SIMP(&a[tid*N], N, P_IND_1);
     SS_REM_PORT(P_IND_1, N, mask, P_IND_1);
   } else {
     SS_DMA_WRITE_SIMP(P_IND_1, N, &b[0]);
     SS_DMA_WRITE_SIMP(P_none_8_B, N, &b[0]);
   }
   SS_WAIT_ALL();

   end_roi();
   sb_stats();
}

void *entry_point(void *threadid) {
  
   long tid;
   tid = (long)threadid;
   // Synchronization point
   int rc = pthread_barrier_wait(&barr);
   if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
   {
     printf("Could not wait on barrier\n");
     // exit(-1);
   }

   remPort(tid);
   // pthread_exit(NULL);
   return NULL;
}


int main(){
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
