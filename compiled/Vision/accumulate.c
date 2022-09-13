#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/i16-128x128x4.h"

#ifndef U
#ifdef FAKE
#define U 4
#else
#define U 16
#endif
#endif

struct Arguments {
  TYPE src[N], dest[N];
  TYPE src_[N], dest_[N];
} args_;

NO_INIT_DATA
NO_SANITY_CHECK

void accumulate(TYPE src[N], TYPE dest[N]){

  #pragma ss config
  {
    arrayhint(src, N * sizeof(TYPE), -1);
    arrayhint(dest, N * sizeof(TYPE), -1);
    int64_t i, j;
    #pragma ss stream nonblock 
    #pragma ss dfg dedicated unroll(U)
    for (i=0; i < N; ++i){
      dest[i] += src[i];
    }
  }
}

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    accumulate(args->src_, args->dest_);
  } else {
    accumulate(args->src, args->dest);
  }
}

