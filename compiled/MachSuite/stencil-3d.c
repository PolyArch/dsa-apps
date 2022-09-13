/*
Implementation based on algorithm described in:
"Stencil computation optimization and auto-tuning on state-of-the-art multicore architectures"
K. Datta, M. Murphy, V. Volkov, S. Williams, J. Carter, L. Oliker, D. Patterson, J. Shalf, K. Yelick
SC 2008
*/

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/stencil-3d.h"

#ifndef U
#define U -1
#endif

void stencil3d(TYPE*__restrict orig, TYPE* __restrict sol) {

  //#pragma ss config
  //{
  //  #pragma ss stream
  //  for(int j=0; j<col_size; j++)
  //    #pragma ss dfg datamove
  //    for(int k=0; k<row_size; k++)
  //      sol[INDX(row_size, col_size, k, j, 0)] = orig[INDX(row_size, col_size, k, j, 0)];
  //}

  //#pragma ss config
  //{
  //  #pragma ss stream
  //  for(int j=0; j<col_size; j++)
  //    #pragma ss dfg datamove
  //    for(int k=0; k<row_size; k++)
  //      sol[INDX(row_size, col_size, k, j, height_size-1)] = orig[INDX(row_size, col_size, k, j, height_size-1)];
  //}

  //#pragma ss config
  //{
  //  #pragma ss stream
  //  for(int i=1; i<height_size-1; i++)
  //    #pragma ss dfg datamove
  //    for(int k=0; k<row_size; k++)
  //      sol[INDX(row_size, col_size, k, 0, i)] = orig[INDX(row_size, col_size, k, 0, i)];
  //}

  //#pragma ss config
  //{
  //  #pragma ss stream
  //  for(int i=1; i<height_size-1; i++)
  //    #pragma ss dfg datamove
  //    for(int k=0; k<row_size; k++)
  //      sol[INDX(row_size, col_size, k, col_size-1, i)] = orig[INDX(row_size, col_size, k, col_size-1, i)];
  //}

  //#pragma ss config
  //{
  //  #pragma ss stream
  //  for(int i=1; i<height_size-1; i++)
  //    #pragma ss dfg datamove
  //    for(int j=1; j<col_size-1; j++)
  //      sol[INDX(row_size, col_size, 0, j, i)] = orig[INDX(row_size, col_size, 0, j, i)];
  //}

  //#pragma ss config
  //{
  //  #pragma ss stream
  //  for(int i=1; i<height_size-1; i++)
  //    #pragma ss dfg datamove
  //    for(int j=1; j<col_size-1; j++)
  //      sol[INDX(row_size, col_size, row_size-1, j, i)] = orig[INDX(row_size, col_size, row_size-1, j, i)];
  //}


  // Stencil computation
  #pragma ss config
  {
    arrayhint(orig, SIZE * sizeof(TYPE), 8.0 / 9.0);
    arrayhint(sol, SIZE * sizeof(TYPE), 0);
    int64_t sum0, sum1, mul0, mul1;
    for(int64_t i = 1; i < height_size - 1; i++) {
      for(int64_t j = 1; j < col_size - 1; j++) {
        #pragma ss stream nonblock
        #pragma ss dfg dedicated unroll(1)
        for(int64_t ko = 1; ko < row_size - 1; ko += 2) {
#define UNROLL_IMPL(k)                                         \
          sum0 = orig[INDX(row_size, col_size, k, j, i)];      \
          sum1 = orig[INDX(row_size, col_size, k, j, i + 1)] + \
                 orig[INDX(row_size, col_size, k, j, i - 1)] + \
                 orig[INDX(row_size, col_size, k, j + 1, i)] + \
                 orig[INDX(row_size, col_size, k, j - 1, i)] + \
                 orig[INDX(row_size, col_size, k + 1, j, i)] + \
                 orig[INDX(row_size, col_size, k - 1, j, i)];  \
          mul0 = sum0 * 12;                                    \
          mul1 = sum1 * 34;                                    \
          sol[INDX(row_size, col_size, k, j, i)] = mul0 + mul1;
          UNROLL_IMPL(ko + 0)
          UNROLL_IMPL(ko + 1)
          // UNROLL_IMPL(ko + 2)
          // UNROLL_IMPL(ko + 3)
        }
      }
    }
  }

}

struct Arguments {
  TYPE a[SIZE], b[SIZE], c[2];
  TYPE a_[SIZE], b_[SIZE], c_[2];
} args_;

NO_INIT_DATA
NO_SANITY_CHECK

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    stencil3d(args->a_, args->b_);
  } else {
    stencil3d(args->a, args->b);
  }
}
