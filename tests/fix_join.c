#include <cassert>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>

#include "testing.h"
#include "join.dfg.h"


int main(int argc, char* argv[]) {
  uint64_t x[6] = {1, 3, 5, 7, 9, SENTINAL};
  uint64_t y[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, SENTINAL};

  SS_CONFIG(join_config, join_size);

  uint64_t answer = 1 + 3 + 5 + 7 + 9;
  uint64_t output = 114514;

  begin_roi(); 
  SS_DMA_READ(&x[0], 8, 8, 6, P_join_A);
  SS_DMA_READ(&y[0], 8, 8, 10, P_join_B);
  SS_DMA_WRITE(P_join_O, 8, 8, 1, &output);
  //SS_RECV(P_join_O, output);
  SS_WAIT_ALL();
  end_roi();

  std::cout << (output - SENTINAL) << ", " << answer << "\n";
  assert((output - SENTINAL) == answer);
  return 0;
}
