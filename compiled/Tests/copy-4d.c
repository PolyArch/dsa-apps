// Test data copy, 4-d memory dimension fusion.

#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#define N 4096

struct Arguments {
  double a[N], b[N], ref[N];
} args_;

void copy4d(double *a, double *b) {
  #pragma ss config
  {
    #pragma ss stream
    for (int64_t i0 = 0; i0 < 8; ++i0) {
      for (int64_t i1 = 0; i1 < 8; ++i1) {
        for (int64_t i2 = 0; i2 < 8; ++i2) {
          #pragma ss dfg
          for (int64_t i3 = 0; i3 < 8; ++i3) {
            int64_t i = i3 + i2 * 8 + i1 * 64 + i0 * 512;
            b[i] = a[i];
          }
        }
      }
    }
  }
}

struct Arguments *init_data() {
  // data initialization
  init_linear(args_.a, N);
  return &args_;
}

void run_reference(struct Arguments *args) {
  double *a = args->a;
  double *ref = args->ref;
  for (int64_t i0 = 0; i0 < 8; ++i0) {
    for (int64_t i1 = 0; i1 < 8; ++i1) {
      for (int64_t i2 = 0; i2 < 8; ++i2) {
        for (int64_t i3 = 0; i3 < 8; ++i3) {
          int64_t i = i3 + i2 * 8 + i1 * 64 + i0 * 512;
          ref[i] = a[i];
        }
      }
    }
  }
}

void run_accelerator(struct Arguments *args) {
  copy4d(args->a, args->b);
}

int sanity_check(struct Arguments *args) {
  compare(args->b, args->ref, N, "%lf");
  return 1;
}

