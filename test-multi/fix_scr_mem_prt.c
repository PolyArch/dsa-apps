#include "testing.h"
#define NUM_THREADS	2

#define N 8

uint16_t a[2*N];
uint16_t b[2*N];
uint16_t c[2*N];
uint16_t d[2*N];
// Barrier variable
pthread_barrier_t barr;

void scr_rem_port_code(long tid) {
   begin_roi();
   SS_CONFIG(none2_config,none2_size);
   if(tid==0){
	 SS_DMA_READ(&a[0], 2, 2, N, P_none2_A);
	 SS_SCR_WRITE(P_none2_B, N*2, 0);
	 SS_WAIT_SCR_WR();
	 SS_SCR_REM_PORT(0, N*2, 2, P_none2_A);
   } else if(tid==1){
	 SS_DMA_WRITE(P_none2_B, 2, 2, N, &b[0]);
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

   scr_rem_port_code(tid);
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

