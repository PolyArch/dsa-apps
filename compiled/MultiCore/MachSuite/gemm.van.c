/*
Implementation based on algorithm described in:
The cache performance and optimizations of blocked algorithms
M. D. Lam, E. E. Rothberg, and M. E. Wolf
ASPLOS 1991
*/

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#include "../Common/multicore.h"
#include "../Specs/gemm.h"


#ifndef U
#define U 4
#endif

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N], int64_t cid){
  int i, k, j, jj, kk = 0;
  int i_row, k_row;
  TYPE temp_x, mul;

  #pragma ss config
  {
    int64_t tc = row_size / block_size;
    int64_t chunk = tc / NUM_CORES;
    for (int64_t jo = 0; jo < chunk; jo++){
      jj = (jo + chunk * cid) * block_size;
      TYPE acc[block_size];
      #pragma ss stream
      for (i = 0; i < row_size; ++i) {
        i_row = i * row_size;
        acc[0] = 0;
        acc[1] = 0;
        acc[2] = 0;
        acc[3] = 0;
        #pragma ss dfg
        for (k = 0; k < row_size; ++k){
          k_row = (k  + kk) * row_size;
          temp_x = m1[i_row + k + kk];
          #define IMPLJ(j) \
          do {             \
            mul = temp_x * m2[k_row + j + jj]; \
            acc[j] += mul; \
          } while (0)
          IMPLJ(0);
          IMPLJ(1);
          IMPLJ(2);
          IMPLJ(3);
          // IMPLJ(4);
          // IMPLJ(5);
          // IMPLJ(6);
          // IMPLJ(7);
        }
        prod[i_row + 0 + jj] += acc[0];
        prod[i_row + 1 + jj] += acc[1];
        prod[i_row + 2 + jj] += acc[2];
        prod[i_row + 3 + jj] += acc[3];
      }
    }
  }
}

// void bbgemm(TYPE a[N], TYPE b[N], TYPE prod[N]) {
//   #pragma ss config
//   {
//     TYPE *m1 = a;
//     TYPE *m2 = b;
// 
//     for (int64_t jo = 0; jo < col_size / 4; jo++){
//       #pragma ss stream nonblock
//       for (int64_t i = 0; i < row_size; ++i){
//         TYPE acc0 = 0;
//         TYPE acc1 = 0;
//         TYPE acc2 = 0;
//         TYPE acc3 = 0;
//         #pragma ss dfg dedicated
//         for (int64_t k = 0; k < col_size; ++k){
//           acc0 += m1[i * col_size + k] * m2[jo * row_size * 4 + k * 4 + 0];
//           acc1 += m1[i * col_size + k] * m2[jo * row_size * 4 + k * 4 + 1];
//           acc2 += m1[i * col_size + k] * m2[jo * row_size * 4 + k * 4 + 2];
//           acc3 += m1[i * col_size + k] * m2[jo * row_size * 4 + k * 4 + 3];
//         }
//         prod[i * col_size + jo * 4 + 0] = acc0;
//         prod[i * col_size + jo * 4 + 1] = acc1;
//         prod[i * col_size + jo * 4 + 2] = acc2;
//         prod[i * col_size + jo * 4 + 3] = acc3;
//       }
//     }
//   }
// }

TYPE a[N], b[N], c[N];
TYPE a_[N], b_[N], c_[N];

void thread_entry(int cid, int nc){
  barrier(nc);
  begin_roi();
  gemm(a, b, c, cid);
  barrier(nc);
  end_roi();
  // return
#ifndef CHIPYARD
  sb_stats();
  if (cid != 0) {
    pthread_exit(NULL);
  }
#else
  exit(0);
#endif
}

