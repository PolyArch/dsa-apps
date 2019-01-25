#include "testing.h"
#include <stdlib.h>

#define VTYPE uint64_t
#define GSIZE 1024*1024*128

int main(int argc, char* argv[]) {
  char* giant_array = (char*) aligned_alloc(64, GSIZE);

  SS_CONFIG(none_config,none_size);

  SS_DMA_READ(&giant_array[(GSIZE/3)*1],8,8,1,P_none_in);
  SS_DMA_READ(&giant_array[(GSIZE/3)*2],8,8,1,P_none_in);

  SS_DMA_WRITE(P_none_out,8,8,1,&giant_array[GSIZE/2]);
  SS_DMA_WRITE(P_none_out,8,8,1,&giant_array[GSIZE-1]);

  SS_WAIT_ALL();

  delete giant_array;
}
