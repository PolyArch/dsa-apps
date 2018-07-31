#include "sb_insts.h"
#include "temporal.dfg.h"
#include "sim_timing.h"

void run() {
  SB_CONFIG(temporal_config, temporal_size);
  SB_CONST(P_temporal_A, 1, 1000);
  SB_GARBAGE(P_temporal_C, 1000);
  SB_GARBAGE(P_temporal_S, 1000);
  SB_GARBAGE(P_temporal_R, 1000);
  SB_WAIT_ALL();
}
