#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Common/spatial_inrin.h"
#define FAKE
#include "../Specs/i16-128x128x4.h"

#ifndef U
#define U 2
#endif

void derivative(const TYPE* src, TYPE* derH, TYPE* derV, TYPE* Ix, TYPE* Iy, TYPE* Ixy) {
#pragma ss config
  {
    arrayhint(src, N * sizeof(TYPE), 0);
    arrayhint(derV, N * sizeof(TYPE), 0);
    TYPE r1;
#pragma ss stream
    for (int64_t r = 1; r < row_size - 1; r++) {
#pragma ss dfg dedicated unroll(4)
      for (int64_t c = 0; c < col_size; c++) {
        r1 = r * col_size;
        TYPE a1 = src[r1 - col_size + c];
        TYPE a2 = src[r1 + c];
        TYPE a3 = src[r1 + col_size + c];
        derV[r1 - col_size + c] = a1 + a2 /*+ a2*/ + a3;
      }
    }
  }

#pragma ss config
  {
    arrayhint(src, N * sizeof(TYPE), 1.0);
    arrayhint(derH, N * sizeof(TYPE), 0);
#pragma ss stream
     for (int64_t r = 0; r < row_size; r++) {
#pragma ss dfg dedicated unroll(1)
       for (int64_t c = 1; c < col_size - 1; c += 2) {
         int64_t rowOffset = r * col_size;
         TYPE a1 = src[rowOffset + c - 1];
         TYPE a2 = src[rowOffset + c];
         TYPE a3 = src[rowOffset + c + 1];
         TYPE a4 = src[rowOffset + c + 2];
         derH[rowOffset + c - 1] = a1 + a2 /*+ a2*/ + a3;
         derH[rowOffset + c] = a2 + a3 /*+ a2*/ + a4;
       }
     }
  }
 
#pragma ss config
  {
    arrayhint(derH, N * sizeof(TYPE), 1.0);
    arrayhint(derV, N * sizeof(TYPE), 1.0);
    arrayhint(Ix, N * sizeof(TYPE), 0);
    arrayhint(Iy, N * sizeof(TYPE), 0);
    arrayhint(Ixy, N * sizeof(TYPE), 0);
#pragma ss stream
     for (int64_t r = 0; r < row_size - 2; r++) {
#pragma ss dfg dedicated unroll(2)
       for (int64_t c = 0; c < col_size - 2; c++) {
         int64_t rowOffset = r * col_size;
         Ix[rowOffset + c] = derH[rowOffset + c] - derH[rowOffset + 2 * col_size + c];
         Iy[rowOffset + c] = derV[rowOffset + c + 2] - derV[rowOffset + c];
         Ixy[rowOffset + c] = Ix[rowOffset + c] * Iy[rowOffset + c];
       }
     }
   }
}

TYPE src[N];
TYPE Ix[N];
TYPE Iy[N];
TYPE Ixy[N];
TYPE derH[N];
TYPE derV[N];

TYPE src_[N];
TYPE Ix_[N];
TYPE Iy_[N];
TYPE Ixy_[N];
TYPE derH_[N];
TYPE derV_[N];

struct Arguments {
} args_;


NO_SANITY_CHECK
NO_INIT_DATA

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    derivative(src_, derH_, derV_, Ix_, Iy_, Ixy_);
  } else {
    derivative(src, derH, derV, Ix, Iy, Ixy);
  }
}
