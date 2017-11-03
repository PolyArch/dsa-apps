#include <stdio.h>
#include "none.h"
#include "check.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

int main(int argc, char* argv[]) {

  begin_roi();
  SB_WAIT_ALL(); 
  end_roi();
}
