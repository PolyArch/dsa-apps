#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/fir.h"


#ifndef U
#define U 4
#endif

void fir(TYPE a[N], TYPE b[M], TYPE c[N - M + 1]) {
  #pragma ss config
  {
    arrayhint(a, N * sizeof(TYPE), 1.0 - (double) N / ((N - M + 1) * M));
    arrayhint(b, M * sizeof(TYPE), 1.0 - 1.0 / (N - M + 1));
    arrayhint(c, (N - M + 1) * sizeof(TYPE), 31.0 / 32);
    TYPE spad_a[N];
    #pragma ss stream nonblock
    #pragma ss dfg unroll(4)
    for (int64_t i = 0; i < N; ++i) {
      spad_a[i] = a[i];
    }
    TYPE spad_b[M];
    #pragma ss stream
    #pragma ss dfg unroll(4)
    for (int64_t i = 0; i < M; ++i) {
      spad_b[i] = b[i];
    }
    for (int64_t io = 0; io < N - M + 1; io += 32) {
      #pragma ss stream
      for (int64_t j = 0; j < M; ++j) {
        #pragma ss dfg dedicated unroll(4)
        for (int64_t ii = 0; ii < 32; ++ii) {
          int64_t i = io + ii;
          c[i] += spad_a[i + j] * spad_b[j];
        }
      }
    }
  }
}

struct Arguments {
} args_;

NO_INIT_DATA
NO_SANITY_CHECK

TYPE a[N], b[M], c[N - M + 1];
TYPE a_[N], b_[M], c_[N - M + 1];


void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    fir(a_, b_, c_);
  } else {
    fir(a, b, c);
  }
}

