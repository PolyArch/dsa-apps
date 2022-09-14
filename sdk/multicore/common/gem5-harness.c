// Test Driver

#include <stdio.h>
#include <pthread.h>

#include "./multicore.h"
#include "./timing.h"
#include "./interface.h"

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

