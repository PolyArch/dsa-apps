#include <stdio.h>
#include <iostream>
#include <pthread.h>
#include <cassert>
#include "none.dfg.h"
#include "none16.dfg.h"
#include "none16_vec.dfg.h"
#include "none2.dfg.h"
#include "check.h"
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>
#include <stdlib.h>

using namespace std;

//top level should define ASIZE

#define DTYPE  uint16_t
#define ABYTES (sizeof(DTYPE)*ASIZE)
#define AWORDS (ABYTES/8)

#define A2WORDS (AWORDS/16)*8

static DTYPE out[ASIZE]; 
static DTYPE in[ASIZE];

void finalfunc() {
  sb_stats();
}

void init() {
  for(int i = 0; i < ASIZE; ++i) {
    in[i] = i;
    out[i] = 0;
  }

  atexit(finalfunc);
}
