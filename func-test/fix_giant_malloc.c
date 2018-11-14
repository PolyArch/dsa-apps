#include "testing.h"

#define VTYPE uint64_t
#define GSIZE 1024*1024*128

int main(int argc, char* argv[]) {
  char* giant_array = (char*) aligned_alloc(64, GSIZE);

  SB_CONFIG(none_config,none_size);

  SB_DMA_READ(&giant_array[(GSIZE/3)*1],8,8,1,P_none_in);
  SB_DMA_READ(&giant_array[(GSIZE/3)*2],8,8,1,P_none_in);

  SB_DMA_WRITE(P_none_out,8,8,1,&giant_array[GSIZE/2]);
  SB_DMA_WRITE(P_none_out,8,8,1,&giant_array[GSIZE-1]);

  SB_WAIT_ALL();

  delete giant_array;
}
