// The first proof of concept

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"
#include "../Specs/i16-128x128x4.h"

struct Arguments {
  TYPE a[N], b[N], c[N];
  TYPE a_[N], b_[N], c_[N];
} args_;

void vecmax(TYPE *a, TYPE *b, TYPE *c) {
  #pragma ss config
  {
    arrayhint(a, N * sizeof(TYPE), 0);
    arrayhint(b, N * sizeof(TYPE), 0);
    arrayhint(c, N * sizeof(TYPE), 0);
    #pragma ss stream
    #pragma ss dfg dedicated unroll(16)
    for (int i = 0; i < N; ++i) {
      c[i] = max16(a[i], b[i]);
    }
  }
}

NO_INIT_DATA
NO_SANITY_CHECK

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    vecmax(args->a_, args->b_, args->c_);
  } else {
    vecmax(args->a, args->b, args->c);
  }
}
