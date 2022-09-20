/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include <stdint.h>
#include <string.h>

#include "common/test.h"
#include "common/timing.h"
#include "common/spatial_inrin.h"

#ifndef N
#define N 496
#endif

#ifndef M
#define M (N * 4)
#endif

#ifndef TYPE
#define TYPE double
#endif

#ifndef U
#define U 1
#endif

void spmv(TYPE *__restrict val, int64_t *__restrict n, int64_t *begin,
          TYPE *__restrict vec, int64_t *__restrict col, TYPE *__restrict out,
          int64_t total) {
  #pragma ss config
  {
    TYPE spad[N];
    #pragma ss stream
    #pragma ss dfg dedicated
    for (int i = 0; i < N; ++i) {
      spad[i] = vec[i];
    }
    // TYPE *spad = vec;
    int64_t k = 0;
    #pragma ss stream nonblock fifo(col,total)
    for (int64_t i = 0; i < N; ++i) {
      TYPE sum = 0;
      #pragma ss dfg dedicated unroll(U)
      for (int64_t j = 0; j < n[i]; ++j){
        sum += val[begin[i] + j] * spad[col[k]];
        ++k;
      }
      out[i] = sum;
    }
  }
}

struct Arguments {
  TYPE vals[N * N], out[N], vec[N];
  int64_t cols[N * N], begin[N + 1], ns[N];
  int64_t total;

  TYPE vals_[N * N], out_[N], vec_[N];
  int64_t cols_[N * N], begin_[N + 1], ns_[N];
  int64_t total_;
} args_;

struct Arguments *init_data() {
  TYPE *vec = args_.vec;
  int64_t *begin = args_.begin;
  int64_t *ns = args_.ns;
  TYPE *vals = args_.vals;
  int64_t *cols = args_.cols;
  int total = 0;
  init_linear(vec, N);
  for (int i = 0; i < N; ++i) {
    begin[i] = total;
    int n = 3 + (rand() % 3 == 0);
    ns[i] = n;
    for (int j = total; j < total + n; ++j) {
      vals[j] = rand();
      cols[j] = rand() % N;
    }
    total += n;
  }
  args_.total = total;
  printf("%d total sparse elements.\n", total);
  memcpy(args_.vals_, args_.vals, sizeof(args_.vals));
  memcpy(args_.out_, args_.out, sizeof(args_.out));
  memcpy(args_.vec_, args_.vec, sizeof(args_.vec));
  memcpy(args_.cols_, args_.cols, sizeof(args_.cols));
  memcpy(args_.begin_, args_.begin, sizeof(args_.begin));
  memcpy(args_.ns_, args_.ns, sizeof(args_.ns));
  args_.total_ = total;
  return &args_;
}

void run_reference(struct Arguments *_) {
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  if (is_warmup) {
    spmv(args->vals_, args->ns_, args->begin_,
         args->vec_, args->cols_, args->out_, args->total_);
  } else {
    spmv(args->vals, args->ns, args->begin, args->vec,
         args->cols, args->out, args->total);
  }
}

int sanity_check(struct Arguments *args) {
  return 1;
}

