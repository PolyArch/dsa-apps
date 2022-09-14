/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include <stdint.h>
#include <string.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/bfs.h"

#ifndef U
#define U 1
#endif

void bfs(int64_t source, int64_t h[N], int64_t e[M], int64_t begin[N], int64_t end[N]) {
  #pragma ss config
  {
    h[source] = 0;
    int64_t flag = 1;
    int64_t curh = 0;
    while (flag) {
      flag = 0;
      int64_t res = curh + 1;
      for (int64_t i = 0; i < N; ++i) {
        if (h[i] == curh) {
          flag = 1;
          int64_t l = begin[i], r = end[i];
          #pragma ss stream
          #pragma ss dfg
          for (int j = l; j < r; ++j) {
            int64_t *ptr = h + e[j];
            *ptr = min64(*ptr, res);
          }
        }
      }
      curh = curh + 1;
    }
    printf("%ld\n", curh);
  }
}

struct Arguments {
  int64_t source;
  int64_t h[N];
  int64_t e[M];
  int64_t begin[N];
  int64_t end[N];
  int64_t source_;
  int64_t h_[N];
  int64_t e_[M];
  int64_t begin_[N];
  int64_t end_[N];
} args_;

struct Arguments *init_data() {
  int64_t total = 0;
  for (int64_t i = 0; i < N; ++i) {
    args_.h_[i] = args_.h[i] = N + 1;
    args_.begin_[i] = args_.begin[i] = total;
    total += rand() % E + 1;
    args_.end_[i] = args_.end[i] = total;
    for (int64_t j = args_.begin[i]; j < args_.end[i]; ++j) {
      args_.e_[j] = args_.e[j] = rand() % N;
    }
  }
  args_.source_ = args_.source = rand() % N;
  return &args_;
}

void run_reference(struct Arguments *_) {
}

void run_accelerator(struct Arguments *args, int is_warmup) {
  if (is_warmup) {
    bfs(args->source_, args->h_, args->e_, args->begin_, args->end);
  } else {
    bfs(args->source, args->h, args->e, args->begin, args->end);
  }
}

int sanity_check(struct Arguments *args) {
  return 1;
}

