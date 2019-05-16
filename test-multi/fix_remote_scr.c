#include <pthread.h>
#include <stdio.h>
#include <iostream>
#include <cassert>
#include "/home/vidushi/ss-stack/riscv-opcodes/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "none.dfg.h"
#define NUM_THREADS	2

using namespace std;
#define N 8

uint16_t a[2*N];
uint16_t b[2*N];
uint16_t c[2*N];
uint16_t d[2*N];
// Barrier variable
pthread_barrier_t barr;

void execute_code(long tid) {

   // begin_roi();
   SS_CONFIG(none_config,none_size);
   if(tid==0){
     // SS_STRIDE(8,8);
     SS_STRIDE(2,2);
	 SS_DMA_READ_SIMP(&a[0], N, P_none_A);
	 // can create it's SIMP also
     SS_REM_SCRATCH(32768, 2, 2, N, P_none_B, 0);
     SS_DMA_WRITE_SIMP(P_none_C, N, &d[0]);
   } else if(tid==1){
     SS_WAIT_DF(N*2,0); // 64-bytes will be written
     SS_SCRATCH_READ(0, N*sizeof(uint16_t), P_none_A);
     SS_STRIDE(2,2);
	 SS_DMA_WRITE_SIMP(P_none_B, N, &b[0]);
     SS_DMA_WRITE_SIMP(P_none_C, N, &c[0]);
   }
   SS_WAIT_ALL();
   
   // end_roi();
   // sb_stats();
}

void final_code(long tid) {
   SS_CONFIG(none_config,none_size);
   if(tid==0){
     // SS_STRIDE(8,8);
     SS_STRIDE(2,2);
     SS_DMA_READ_SIMP(&a[tid*N], N, P_none_A);
	 // can create it's SIMP also
     SS_REM_SCRATCH(32768, 2, 2, N, P_none_B, 0);
     SS_DMA_WRITE_SIMP(P_none_C, N, &d[0]);
   } else if(tid==1){
     SS_WAIT_DF(N*2,0); // 64-bytes will be written
     SS_SCRATCH_READ(0, N*sizeof(uint16_t), P_none_A);
     SS_STRIDE(2,2);
	 SS_DMA_WRITE_SIMP(P_none_B, N, &b[0]);
     SS_DMA_WRITE_SIMP(P_none_C, N, &c[0]);
   }
   SS_WAIT_ALL();
 
}

// how to make sure that output dest core is 1? (1<<16)
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

   begin_roi();
   execute_code(tid);
   // int rc2 = pthread_barrier_wait(&barr);
   // if(rc2 != 0 && rc2 != PTHREAD_BARRIER_SERIAL_THREAD)
   // {
   //   printf("Could not wait on barrier\n");
   //   // exit(-1);
   // }
   // final_code(tid);
   end_roi();
   sb_stats();
   // pthread_exit(NULL);
   return NULL;
}


int main(){
  // What should changing this matter?
  for(uint16_t i=0; i<N; i++){
    a[i] = i;
    b[i] = i;
    c[i] = i;
    d[i] = i;
  }
  assert(NUM_THREADS<C);
  
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
	// it should diverge instead of being done serially with the host thread?:
	// put a barrier
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
  // Last thing that main() should do 
  // only if threads are created, otherwise gives VA 0 fault
  // pthread_exit(NULL);
   
  return 0;
};

/*
void execute_code(long tid) {

   begin_roi();
   SS_CONFIG(none_config,none_size);
   SS_STRIDE(8,8);
   if(tid==0){
     SS_DMA_READ_SIMP(&a[tid*N], N, P_none_A);
     // SS_IND_REM_SCRATCH(P_none_B, P_none_C, N, 0, 0); // 0 means banked!
     // SS_REM_SCRATCH(scr_base_addr, stride, access_size, num_strides, val_port, scratch_type)
	 // This is direct, remove 0 (this doesn't mean anything in the simulator now)
     SS_REM_SCRATCH(32768, 8, 8, N, P_none_B, 0); // TODO: need to understand this encoding properly
     // SS_REM_SCRATCH(0, 8, 8, N, P_none_B, 0); // TODO: need to understand this encoding properly
     SS_DMA_WRITE_SIMP(P_none_C, N, &d[0]);
   } else if(tid==1){
	 // SS_WAIT_DF(N,0);
	 SS_WAIT_DF(N*8,0);
	 SS_SCRATCH_READ(0, N*sizeof(uint16_t), P_none_A);
     SS_DMA_WRITE_SIMP(P_none_B, N, &b[0]);
     SS_DMA_WRITE_SIMP(P_none_C, N, &c[0]);
   }
   SS_WAIT_ALL();
   
   end_roi();
   sb_stats();
}
*/
