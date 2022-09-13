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
#include "../Specs/gemm.h"


#ifndef U
#define U 4
#endif

void gemm(TYPE a[N], TYPE b[N], TYPE c[N]) {
#define PART 8
#define CHUNK (col_size / T / PART)
#define T 2
  #pragma ss config
  {
    // arrayhint(a, N * sizeof(TYPE), 1 - (1.0 / CHUNK));
    // arrayhint(b, N * sizeof(TYPE), 1 - 1.0 / col_size);
    // arrayhint(c, N * sizeof(TYPE), 1 - (1.0 / (col_size / T)));
    TYPE spad[N];
    #pragma ss stream
    #pragma ss dfg unroll(4)
    for (int64_t i = 0; i < N; ++i) {
      spad[i] = b[i];
    }
    for (int64_t jo = 0; jo < PART; ++jo) {
      for (int64_t i = 0; i < row_size / T; ++i) {
        #pragma ss stream nonblock
        for (int64_t k = 0; k < col_size / T; ++k) {
          #pragma ss dfg
          for (int64_t ji = 0; ji < CHUNK; ++ji) {
            int64_t j = jo * CHUNK + ji;
            #define decl_load(array, idx) TYPE v##array##idx = v##array[idx]
            TYPE *va = &a[i * ((col_size / T) * T * T) + k * T * T]; // [i, k]
            decl_load(a, 0); decl_load(a, 1); decl_load(a, 2); decl_load(a, 3);
            TYPE *vb = &spad[k * ((col_size / T) * T * T) + j * T * T]; // [j, k]
            decl_load(b, 0); decl_load(b, 1); decl_load(b, 2); decl_load(b, 3);
            TYPE *vc = &c[i * ((col_size / T) * T * T) + j * T * T]; // [i, j]
            decl_load(c, 0); decl_load(c, 1); decl_load(c, 2); decl_load(c, 3);
            vc[0] += va0 * vb0 + va1 * vb2;
            vc[1] += va0 * vb1 + va1 * vb3;
            vc[2] += va2 * vb0 + va3 * vb2;
            vc[3] += va2 * vb1 + va3 * vb3;
          }
        }
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

struct Arguments {
  TYPE a[N], b[N], c[N];
  TYPE a_[N], b_[N], c_[N];
} args_;

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    gemm(args->a_, args->b_, args->c_);
  } else {
    gemm(args->a, args->b, args->c);
  }
}

NO_SANITY_CHECK
NO_INIT_DATA
