#include "testing.h"
#include <cassert>

#include "sacc.dfg.h"

#define N 4

int main(int argc, char* argv[]) {

  int64_t res(0);

  SS_CONFIG(sacc_config, sacc_size);
  SS_CONST(P_sacc_a, 1, N);
  SS_CONST(P_sacc_b, 2, N);
  SS_CONST(P_sacc_signal, 2, N - 1);
  SS_CONST(P_sacc_signal, 1, 1);
  SS_DMA_WRITE(P_sacc_d, 8, 8, 1, &res);
  SS_WAIT_ALL();

  assert(res == N * 2);

  return 0;
}
