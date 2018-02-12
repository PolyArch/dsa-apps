#include <complex>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "sb_insts.h"
#include "sim_timing.h"
#include "compute.dfg.h"
#include "subadd.dfg.h"

using std::complex;

void fft(complex<float> *a, complex<float> *w) {
  SB_CONFIG(compute_config, compute_size);

  int N = _N_;
  int span = N >> 1, blocks = 1;
  for ( ; span > 1; span >>= 1, blocks <<= 1) {
    SB_DMA_READ(a, 16 * span, 8 * span, N / span / 2, P_compute_L);
    SB_DMA_READ(a + span, 16 * span, 8 * span, N / span / 2, P_compute_R);
    SB_DMA_WRITE(P_compute_A, 16 * span, 8 * span, N / span / 2, a);
    SB_DMA_WRITE(P_compute_B, 16 * span, 8 * span, N / span / 2, a + span);
    for (int j = 0; j < blocks; ++j) {
      SB_DMA_READ(w, blocks * 8, 8, span, P_compute_W);
    }
    SB_WAIT_ALL();
    //for (int j = 0; j < N; ++j)
      //std::cout << a[j] << (j == N - 1 ? "\n" : " ");
    /*
    for (int odd = span, even; odd < N; ++odd) {
      odd |= span;
      even = odd ^ span;

      complex<float> temp = a[even] + a[odd];
      a[odd]  = a[even] - a[odd];
      a[even] = temp;

      int index = (even << _log) & (N - 1);
      if (index) {
        a[odd] *= w[index];
        //printf("[%d] %d\n", span, index);
      }
    }
    */
  }

  SB_CONFIG(subadd_config, subadd_size);
  SB_DMA_READ(a, 16 * span, 8 * span, N / span / 2, P_subadd_L);
  SB_DMA_READ(a + span, 16 * span, 8 * span, N / span / 2, P_subadd_R);
  SB_DMA_WRITE(P_subadd_A, 16 * span, 8 * span, N / span / 2, a);
  SB_DMA_WRITE(P_subadd_B, 16 * span, 8 * span, N / span / 2, a + span);
  SB_WAIT_ALL();

}
