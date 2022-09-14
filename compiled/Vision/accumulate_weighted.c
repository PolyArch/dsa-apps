#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/i16-128x128x4.h"


#ifndef U
#define U 4
#endif

struct Arguments {
  TYPE src[N], dest[N];
  TYPE src_[N], dest_[N];
  TYPE temp;
  TYPE temp1;
} args_;

void accumulate_weighted(TYPE src[N], TYPE dest[N], TYPE temp1, TYPE temp2, TYPE temp3){
  #pragma ss config
  {
    arrayhint(src, N * sizeof(TYPE), 0);
    arrayhint(dest, N * sizeof(TYPE), 0);
    int64_t i;
    #pragma ss stream 
    #pragma ss dfg dedicated unroll(4)
    for (i = 0; i < N; ++i){
      dest[i] = div16((src[i] * 1234 + dest[i] * 4321), 3456);
    }
  }
}

struct Arguments *init_data() {
  args_.temp = 1234;
  args_.temp1 = 4321;
  return &args_;
}

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    accumulate_weighted(args->src_, args->dest_, 1234, 4321, 3456);
  } else {
    accumulate_weighted(args->src, args->dest, 1234, 4321, 3456);
  }
}

NO_SANITY_CHECK
