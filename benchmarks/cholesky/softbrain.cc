#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "compute.h"
#include "writeback.h"
#include "sb_insts.h"

using std::complex;

struct complex_t {
  float real, imag;
};

void cholesky(complex<float> *a, complex<float> *L) {

  complex_t *A = new complex_t[N * N];
  for (int i = 0; i < N * N; ++i) {
    A[i].real = a[i].real();
    A[i].imag = a[i].imag();
  }
  for (int i = 0; i < N; ++i) {
    complex<float> _div = std::sqrt(complex<float>(A[i * (N + 1)].real, A[i * (N + 1)].imag));
    complex_t div = {_div.real(), _div.imag()};

    complex_t *b = A + i * (N + 1);
    L[i * (N + 1)] = std::complex<float>(div.real, div.imag);
    {
      complex_t *bp = b + 1;
      float norm = 1 / ((div.real * div.real) + (div.imag * div.imag));
      union {
        float f[2];
        uint64_t v;
      } ri_norm = {norm, norm};
      SB_CONFIG(writeback_config, writeback_size);
      SB_DMA_READ(bp, 8, 8, N - i - 1, P_writeback_BP);
      SB_CONST(P_writeback_NORM, ri_norm.v, N - i - 1);
      SB_CONST(P_writeback_DIV, *((uint64_t *) &div), N - i - 1);
      SB_DMA_WRITE(P_writeback_RES, N * 8, 8, N - i - 1, L + (i + 1) * N + i);
      /*for (int j = i + 1; j < N; ++j) {
        //L[j * N + i] = b[j - i] / div;
        L[j * N + i] = complex<float>(
            (bp->real * div.real + bp->imag * div.imag) * norm,
            (bp->imag * div.real - bp->real * div.imag) * norm
        );
        ++bp;
      }*/
    }
    {
      complex_t *bj = b + 1, *bk, v = A[i * (N + 1)];
      float norm = 1 / (v.real * v.real + v.imag * v.imag);
      union {
        float f[2];
        uint64_t v;
      } ri_norm = {-norm, -norm}, ri_v = {v.real, v.imag};
      SB_WAIT_ALL();
      SB_CONFIG(compute_config, compute_size);


      for (int j = i + 1; j < N; ++j) {
        int times = N - j;
        uint64_t ri_bj = *((uint64_t *) bj);

        SB_CONST(P_compute_A, ri_bj, times);
        SB_DMA_READ(A + j * (N + 1), 0, 8 * times, 1, P_compute_Z);
        SB_DMA_READ(bj, 0, 8 * times, 1, P_compute_B);
        SB_CONST(P_compute_NORM, ri_norm.v, times);
        SB_CONST(P_compute_V, ri_v.v, times);
        SB_DMA_WRITE(P_compute_O, 0, 8 * times, 1, A + j * (N + 1));

        ++bj;
      }
      SB_WAIT_ALL();
    }
  }
  delete []A;
}

