#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "compute.dfg.h"
#include "writeback.dfg.h"
#include "sb_insts.h"
#include "matvec.h"

using std::complex;

void cholesky(complex<float> *a, complex<float> *L) {

  for (int i = 0; i < N; ++i) {
    complex<float> div = std::sqrt(a[i * N + i]);

    complex<float> *b = a + i * (N + 1);
    L[i * (N + 1)] = div;
    {
      complex<float> *bp = b + 1;
      float norm = 1 / complex_norm(div);
      div = std::conj(div * norm);
      SB_CONFIG(writeback_config, writeback_size);
      SB_DMA_READ(bp, 8, 8, N - i - 1, P_writeback_BP);
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
      complex<float> *bj = b + 1, *bk, v = a[i * (N + 1)];
      float norm = 1 / complex_norm(v);
      union {
        float f[2];
        uint64_t v;
      } ri_v = {v.real() * norm, v.imag() * -norm};
      SB_WAIT_ALL();
      SB_CONFIG(compute_config, compute_size);


      for (int j = i + 1; j < N; ++j) {
        int times = N - j;
        uint64_t ri_bj = *((uint64_t *) bj);

        SB_CONST(P_compute_A, ri_bj, times);
        SB_DMA_READ(a + j * (N + 1), 0, 8 * times, 1, P_compute_Z);
        SB_DMA_READ(bj, 0, 8 * times, 1, P_compute_B);
        SB_CONST(P_compute_V, ri_v.v, times);
        SB_DMA_WRITE(P_compute_O, 0, 8 * times, 1, a + j * (N + 1));

        ++bj;
      }
      SB_WAIT_ALL();
    }
  }
}

