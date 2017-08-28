#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "compute.h"
#include "sb_insts.h"

void cholesky(float *a, float *L) {
  const int vec_bytes = VEC_WIDTH * sizeof(float);
  const int n = (N + VEC_WIDTH) | 1;
  float *A = (float *) malloc(n * n * sizeof(float));

  SB_CONFIG(compute_config, compute_size);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * n + j] = a[i * N + j];
    }
  }
  for (int i = 0; i < N; ++i) {
    float div = sqrt(A[i * (n + 1)]);
    //float *b = A + i * (n + 1);
    float b[N - i + (VEC_WIDTH - 1)];
    float bb[N - i - 1 + (VEC_WIDTH - 1)];
    L[i * (N + 1)] = div;
    {
      float *_b = A + i * (n + 1);
      float *bp = b;
      *bp++ = *_b++;
      float *bbp = bb;
      for (int j = i + 1; j < N; ++j) {
        float v = *_b++;
        *bp++ = v;
        *bbp++= v;
        L[j * N + i] = v / div;
      }
      for (int j = 1; j < VEC_WIDTH; ++j)
        *bp++ = *bbp++ = 0;
    }
    //begin_roi();
    {
      float *bj = b + 1, *bk;
      //float v = A[i * (n + 1)];
      float v = 1. / A[i * (n + 1)];
      char delta = 0;
      for (int j = i + 1; j < N; ++j) {

        union {
          float a[2];
          uint64_t v;
        } ri_bj = {*bj, *bj}, ri_v = {-v, -v};

        int rw = (N - j - 1) / VEC_WIDTH + 1;
        int cnt= rw * VEC_WIDTH / 2;
        //assert(rw == cnt);
        
        delta ^= 1;
        SB_CONST(P_compute_A, ri_bj.v, cnt);
        SB_DMA_READ(delta ? bb + j - i - 1 : b + j - i, vec_bytes, vec_bytes, rw, P_compute_B);
        SB_CONST(P_compute_C, ri_v.v, cnt);
        SB_DMA_READ(A + (n + 1) * j, vec_bytes, vec_bytes, rw, P_compute_D);
        SB_DMA_WRITE(P_compute_G, vec_bytes, vec_bytes, rw, A + (n + 1) * j);

        /*
        bk = b + j - i;
        for (int k = j; k < N; ++k) {
          A[j * n + k] -= *bj * (*bk++) * v;
        }*/
        ++bj;
      }
      SB_WAIT_ALL();
    }
    //end_roi();
  }
}

