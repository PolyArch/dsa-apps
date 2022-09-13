// Test Driver
#include <stdint.h>
#include <stdio.h>

#include "../Common/interface.h"
#include <sys/time.h>

static uint64_t ticks;

static __inline__ uint64_t get_time(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}


int main() {
  int64_t ref = 0;
  struct Arguments *args = init_data();
  printf("[single-core] initialization finished\n");
  // 1st run
  {
    int64_t start = get_time();
    run_accelerator(args, 1);
    printf("[single-core] warm i-cache finished, %ld us passed! \n", get_time() - start);
  }
  // 2nd run
  {
    int64_t start = get_time();
    run_accelerator(args, 0);
    printf("[single-core] Ticks: %ld us\n", get_time() - start);
  }
  return 0;
}

