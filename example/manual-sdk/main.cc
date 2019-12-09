#include <iostream>
#include <ss-intrin/ss_insts.h>
#include <sim_timing.h>
#include "dfg.dfg.h"

uint64_t a[100], b[100], c[100];

void foo() {
  SS_CONFIG(dfg_config, dfg_size);
  SS_DMA_READ(a, 8, 8, 100, P_dfg_a);
  SS_DMA_READ(b, 8, 8, 100, P_dfg_b);
  SS_DMA_WRITE(P_dfg_c, 8, 8, 100, c);
  SS_WAIT_ALL();
}

int main() {
  foo();
  begin_roi();
  foo();
  end_roi();
  sb_stats();
  return 0;
}
