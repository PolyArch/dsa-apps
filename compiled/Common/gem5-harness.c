// Test Driver

#include "../Common/timing.h"
#include "../Common/interface.h"


#ifndef NUM_CORES

int main() {
  int64_t ref = 0;
  struct Arguments *args = init_data();
  printf("[single-core] initialization finished\n");
  uint64_t start = rdcycle();
  run_reference(args);
  printf("[single-core] cpu pass finished, %lu cycles passed! \n", rdcycle() - start);
  // 1st run
  start = rdcycle();
  run_accelerator(args, 1);
  printf("[single-core] warm i-cache finished, %lu cycles passed! \n", rdcycle() - start);
  // 2nd run
  begin_roi();
  run_accelerator(args, 0);
  end_roi();
  sb_stats();
  // sanity check
  printf("[single-core] accelerator finished ...\n");
  if(sanity_check(args)){
    printf("[single-core] sanity check passed successfully! \n");
    return 0;
  }else{
    printf("[single-core] sanity check did not pass!\n");
    return 1;
  }
}

#else

#include <stdio.h>
#include <pthread.h>

#include "../Common/multicore.h"

pthread_barrier_t pb;

void barrier(int nc) {
  asm volatile("fence");
  pthread_barrier_wait(&pb);
}

void thread_entry(int, int);

void *exec(void *args) {
  thread_entry((int64_t) args, NUM_CORES);
  return NULL;
}

int main(int argc, char **argv) {
  pthread_barrier_init(&pb, NULL, NUM_CORES);
  pthread_t pthr[NUM_CORES];
  for (int64_t i = 1; i < NUM_CORES; ++i) {
    pthread_create(pthr + i, NULL, exec, (void*) i);
  }
  exec(0);
  for (int64_t i = 1; i < NUM_CORES; ++i) {
    void *res;
    pthread_join(pthr[i], &res);
    int64_t val = (int64_t)(res);
    if (val != 0) {
      return 1;
    }
  }
  printf("[multicore] check pass!\n");
  return 0;
}

#endif
