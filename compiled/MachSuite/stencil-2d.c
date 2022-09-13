#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/stencil-2d.h"

void stencil(TYPE orig[N * N], TYPE sol[N * N], TYPE filter[9]) {
  #pragma ss config
  {
    int r, c, k1, k2;
    TYPE mul;
    for (r = 0; r < N - 2; r++) {
      #pragma ss stream nonblock
      #pragma ss dfg dedicated // unroll(4)
      for (c = 0; c < N - 2; c++) {
        sol[r * N + c] = orig[(r + 0) * N + c + 0] * filter[k1 * 3 + 0] +
                         orig[(r + 0) * N + c + 1] * filter[k1 * 3 + 1] +
                         orig[(r + 0) * N + c + 2] * filter[k1 * 3 + 2] +
                         orig[(r + 1) * N + c + 0] * filter[k1 * 3 + 0] +
                         orig[(r + 1) * N + c + 1] * filter[k1 * 3 + 1] +
                         orig[(r + 1) * N + c + 2] * filter[k1 * 3 + 2] +
                         orig[(r + 2) * N + c + 0] * filter[k1 * 3 + 0] +
                         orig[(r + 2) * N + c + 1] * filter[k1 * 3 + 1] +
                         orig[(r + 2) * N + c + 2] * filter[k1 * 3 + 2];
      }
    }
  }
}

struct Arguments {
  int64_t a[N * N], b[N * N], c[9];
  int64_t a_[N * N], b_[N * N], c_[9];
} args_;

NO_SANITY_CHECK
NO_INIT_DATA

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    stencil(args->a, args->b, args->c);
  } else {
    stencil(args->a_, args->b_, args->c_);
  }
}
