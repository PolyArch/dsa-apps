#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS	3

void *PrintHello(void *threadid)
{
   long tid;
   tid = (long)threadid;
   printf("Hello World! It's me, thread #%ld!\n", tid);
   pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
   pthread_t threads[NUM_THREADS];
   int rc;
   long t;
   for(t=0;t<NUM_THREADS;t++){
     printf("In main: creating thread %ld\n", t);
     rc = pthread_create(&threads[t], NULL, PrintHello, (void *)t);
     if (rc){
       printf("ERROR; return code from pthread_create() is %d\n", rc);
       exit(-1);
       }
     }

   // Last thing that main() should do 
   pthread_exit(NULL);
}

/*

#include <stdio.h>
#include <omp.h>
#include "inttypes.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"

#define N 3

int main(){


  omp_set_num_threads(C);
  printf("number of cores: %d\n",C);

  int tid = 0;
  #pragma omp parallel private(tid) // create copies in the local memory (not a shared variable anymore)
  {
	tid =  omp_get_thread_num();
	printf("hello world!!! %d\n",tid);
  }

  uint64_t a[N];
  
  for(uint64_t i=0; i<N; i++){
    a[i] = i;
  }

  #pragma omp parallel for // private(i) num_threads(C): keep loop variable private
  for(uint64_t i=0; i<N; i++){
    a[i] += 2;
  }
  

  return 0;
};
*/
