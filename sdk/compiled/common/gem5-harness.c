// Test Driver

#include "./timing.h"
#include "./interface.h"

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

