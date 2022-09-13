#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#define FAKE
#include "../Specs/i16-128x128x4.h"

void channelExtract(const TYPE* src, TYPE* dest, int64_t channel){
  #pragma ss config
  {
    arrayhint(src, N * C * sizeof(TYPE), 0);
    arrayhint(dest, N * sizeof(TYPE), 0);
    int64_t i, j;
    #pragma ss stream 
    for(i = 0; i< row_size; ++i){
      #pragma ss dfg dedicated unroll (1)
      for(j = 0; j < col_size; ++j){
        dest[i * col_size + j] = src[i * col_size * C + j * C + channel];
      }
    }
  }
}


struct Arguments {
  TYPE src_[N * C], dest_[N];
  TYPE src[N * C], dest[N];
} args_;

struct Arguments *init_data() {
  return &args_;
}

void run_reference(struct Arguments *args) {
}

void run_accelerator(struct Arguments *args, int iswarmup) {
  if (iswarmup) {
    channelExtract(args->src_, args->dest_, 0);
  } else {
    channelExtract(args->src, args->dest, 0);
  }
}

int sanity_check(struct Arguments *args) {
  return 1;
}
