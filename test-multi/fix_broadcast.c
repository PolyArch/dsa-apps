#include <assert.h>
#include "testing.h"
#include "none.dfg.h"

#define NUM_THREADS 4

#define N 4096
#define T 2
#define VTYPE uint64_t
int c=1;

VTYPE a[N];
VTYPE b[NUM_THREADS*N];

// Barrier variable
pthread_barrier_t barr;

// read from memory->core with tid=0, then broadcast to all other cores
void remPort(long tid) {

  uint64_t mask=0;
  for(int i=1; i<NUM_THREADS; ++i) addDest(mask,i);

   if(c==T) begin_roi();
   SS_CONFIG(none_config,none_size);
   if(tid==0){
     SS_DMA_READ(&a[0], 8, 8, N, P_none_in);
     SS_REM_PORT(P_none_out, N, mask, P_none_in);
   } else {
     SS_DMA_WRITE(P_none_out, 8, 8, N, &b[tid*N]);
   }
   SS_WAIT_ALL();
   // SS_GLOBAL_WAIT(); // this will give approx equal cycles

   if(c==T) {
     end_roi();
     sb_stats();
   }
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
   return NULL;
}


int main(){
  // data generation
  for(int i=0; i<N; i++){
    a[i] = i;
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
  

  for(int r=1; r<=T; ++r) {
    c=r;
    for(t=0;t<NUM_THREADS;t++){
      printf("In main: creating thread %ld\n", t);
      rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);
      if (rc){
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        return 0;
      }
    }
     
    // this is a barrier or not?
    for(int i = 0; i < NUM_THREADS; ++i) {
      if(pthread_join(threads[i], NULL)) {
    	printf("Could not join thread %d\n", i);
        return -1;
      }
    }
  }

  return 0;
};
