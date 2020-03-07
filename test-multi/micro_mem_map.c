#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "none.dfg.h"

#define NUM_THREADS 2
pthread_barrier_t barr;

#define LEN 256
#define SLEN 4 // 64
  
uint64_t ind_array[SLEN]; //16-bit ind type

uint64_t array[SLEN]; //32-bit data type
uint64_t known[SLEN]; //32-bit data type
uint64_t output[SLEN];
 

void indRead(long tid) {

  begin_roi();
  if(tid==0) {


  //lets copy array to scratch
  // SS_DMA_SCRATCH_LOAD(array,8,8,SLEN,0);
  // SS_WAIT_SCR_WR();

    SS_CONFIG(none_config,none_size);


    for(int i = 0; i < 1; ++ i) {
      SS_DMA_READ(&ind_array[0],8,8,SLEN,P_IND_1);
    
      // part_size=8*2 bytes, core_bv=11, map_type=0
      SS_CONFIG_MEM_MAP(16,3,0);
      //itype, dtype, mult, offset
      SS_CONFIG_INDIRECT(T64,T64,8,1);
      SS_INDIRECT_SCR(P_IND_1,0,SLEN,P_none_in); // 0, 8, 16, 32
    
      SS_DMA_WRITE(P_none_out,8,8,SLEN,&output[0]);
    }
    SS_WAIT_ALL();
  }
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
   indRead(tid);
   // pthread_exit(NULL);
   return NULL;
}


int main(){

  for(int i = 0; i < SLEN; ++i) {
    ind_array[i] = i; // rand()%SLEN; // 0,1,2,4
    array[i] = i;
  }


  for(int i = 0; i<SLEN; ++i) {
    int ind = ind_array[i];

    known[i]=array[ind];
    known[i]=array[ind];
    known[i]=array[ind];

    output[i]=-1;
    output[i]=-1;
    output[i]=-1;
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
