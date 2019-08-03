#include "testing.h"
#define NUM_THREADS	2

#define VTYPE uint64_t

#define N 64

VTYPE a[2*N];
VTYPE b[2*N];
VTYPE c[2*N];
VTYPE d[2*N];
// Barrier variable
pthread_barrier_t barr;

// copy from a port to a global scratchpad location
void execute_code(long tid) {
  int d = sizeof(VTYPE);

   begin_roi();
   SS_CONFIG(none_config,none_size);
   if(tid==0){
	 SS_DMA_READ(&a[0], d, d, N, P_none_in);
     SS_REM_SCRATCH(32768, d, d, N, P_none_out, 0); // Why is this not empty? or this is empty earlier?
   } else if(tid==1){
     SS_WAIT_DF(N*d,0); // 64-bytes will be written 
     // so it should wait on 128*8 = 1024 bytes (=904/64 written = 1...)
     SS_SCRATCH_READ(0, N*d, P_none_in);
	 SS_DMA_WRITE(P_none_out, d, d, N, &b[0]); // error- how is this issued when barrier is not yet released?
   }
   SS_WAIT_ALL();
   
   end_roi();
   sb_stats();
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

   execute_code(tid);
   return NULL;
}


int main(){
  // What should changing this matter?
  for(VTYPE i=0; i<N; i++){
    a[i] = i;
    b[i] = i;
    c[i] = i;
    d[i] = i;
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
   
  return 0;
};
