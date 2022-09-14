#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/i16-128x128x4.h"

#ifndef U
#define U 16
#endif

void accumulate_squared(TYPE src[N], TYPE dest[N]){

  #pragma ss config
  {
    arrayhint(src, N * sizeof(TYPE), 0);
    arrayhint(dest, N * sizeof(TYPE), 0);
    int64_t i;
    #pragma ss stream nonblock 
    #pragma ss dfg dedicated unroll(U)
    for (i = 0; i < N; ++i){
      dest[i] += src[i] * src[i];
    }
  }
}

struct Arguments {
  TYPE src[N], dest[N];
  TYPE src_[N], dest_[N];
} args_;

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    accumulate_squared(args->src_, args->dest_);
  } else {
    accumulate_squared(args->src, args->dest);
  }
}

NO_INIT_DATA
NO_SANITY_CHECK

