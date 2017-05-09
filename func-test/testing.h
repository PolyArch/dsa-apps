#include <stdio.h>
#include "none.h"
#include "check.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#include <inttypes.h>

//top level shoudl define ASIZE

#define DTYPE  uint16_t
#define ABYTES (sizeof(DTYPE)*ASIZE)
#define AWORDS (ABYTES/8)

#define A2WORDS (AWORDS/16)*8

static DTYPE out[ASIZE]; 
static DTYPE in[ASIZE];
static DTYPE mod[ASIZE];
static DTYPE str2[ASIZE];


void init() {
  for(int i=0; i < ASIZE; ++i) {
    in[i]=i;
    out[i]=0;
    mod[i]=i%4;
  }
  for(int i = 0; i < A2WORDS/sizeof(DTYPE)*8; i+=1) {
    str2[i]=(i/4)*8+i%4;
  }
}
