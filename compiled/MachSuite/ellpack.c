#include <stdint.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/ellpack.h"


#ifndef U
#define U 4
#endif

void ellpack(TYPE *__restrict nzval, int64_t *__restrict cols,
             TYPE *__restrict vec, TYPE *__restrict out) {
  #pragma ss config
  {
    arrayhint(nzval, N * L * sizeof(TYPE), 0);
    arrayhint(cols, N * L * sizeof(TYPE), 0);
    arrayhint(vec, N * sizeof(TYPE), 1 - 1.0 / N);
    arrayhint(out, N * sizeof(TYPE), 0.75);

    TYPE spad[N];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int i = 0; i < N; ++i) {
      spad[i] = vec[i];
    }

    #pragma ss stream
    for (int i = 0; i < N; ++i) {
      TYPE sum = 0.0;
      #pragma ss dfg dedicated unroll(4)
      for (int j = 0; j < L; ++j) {
        sum += nzval[j + i * L] * spad[cols[j + i * L]];
      }
      out[i] = sum;
    }
  }
}

NO_SANITY_CHECK

struct Arguments {
  TYPE val[N * L], vec[N], out[N];
  int64_t cols[N * L];

  TYPE val_[N * L], vec_[N], out_[N];
  int64_t cols_[N * L];
}args_;

struct Arguments *init_data() {
  TYPE *val = args_.val;
  int64_t *cols = args_.cols;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < L; ++j) {
      val[i * L + j] = rand();
      cols[i * L + j] = rand() % N;
    }
  }
  return &args_;
}


void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    ellpack(args->val, args->cols, args->vec, args->out);
  } else {
    ellpack(args->val_, args->cols_, args->vec_, args->out_);
  }
}
