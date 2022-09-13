#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/mm.h"


#ifndef U
#define U 4
#endif

void mm(TYPE *a, TYPE *b, TYPE *c) {
  #pragma ss config
  {
    arrayhint(a, N * M * sizeof(TYPE), 0);
    arrayhint(b, M * M * sizeof(TYPE), 1.0 - 1.0 / (M * P));
    arrayhint(c, N * P * sizeof(TYPE), 0);
    TYPE bb[M * P];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for (int i = 0; i < M * P; ++i)
      bb[i] = b[i];
    for (int i = 0; i < N; ++i) {
      #pragma ss stream nonblock
      for (int k = 0; k < M; ++k) {
        #pragma ss dfg dedicated unroll(4)
        for (int j = 0; j < P; ++j) {
          c[i * P + j] += a[i * M + k] * bb[k * P + j];
          // c[i * P + j] += a[i * M + k] * b[k * P + j];
        }
      }
    }
  }
}

struct Arguments {
  TYPE a[N * M], b[M * P], c[N * P];
  TYPE a_[N * M], b_[M * P], c_[N * P];
} args_;

struct Arguments *init_data() {
  return &args_;
}

void run_reference(struct Arguments *_) {
}

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    mm(args->a_, args->b_, args->c_);
  } else {
    mm(args->a, args->b, args->c);
  }
}

int sanity_check(struct Arguments *_) {
  return 1;
}
