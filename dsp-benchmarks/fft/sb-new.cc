#include <complex>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "sb_insts.h"
#include "sim_timing.h"
#include "rotate.dfg.h"
#include "subadd.dfg.h"

using std::complex;

void fft(complex<float> *a, complex<float> *w) {
  SB_CONFIG(rotate_config, rotate_size);

  int N = _N_;
  int span = N >> 1, blocks = 1;

  {
    SB_DMA_READ(a, 16 * span, 8 * span, N / span / 2, P_rotate_L);
    SB_DMA_READ(a + span, 16 * span, 8 * span, N / span / 2, P_rotate_R);
    SB_SCR_WRITE(P_rotate_A, N * 4, 0);
    SB_SCR_WRITE(P_rotate_B, N * 4, N * 4);

    SB_CONST(P_rotate_W, 1065353216, 1);
    SB_CONST(P_rotate_Rotate, *((uint64_t*)(w + 1)), N / 2 - 1);
    SB_CONST(P_rotate_Current, 1065353216, 1);
    SB_RECURRENCE(P_rotate_Angle, P_rotate_W, N / 2 - 1);
    SB_RECURRENCE(P_rotate_Rec, P_rotate_Current, N / 2 - 2);
    SB_GARBAGE(P_rotate_Rec, 1);

    //for (int j = 0; j < blocks; ++j) {
      //SB_DMA_READ(w, blocks * 8, 8, span, P_rotate_W);
    //}
    SB_WAIT_SCR_WR();
    span >>= 1;
    blocks <<= 1;
  }

  for ( ; span > 1; span >>= 1, blocks <<= 1) {
    SB_SCR_PORT_STREAM(0       , 16 * span, 8 * span, N / span / 2, P_rotate_L);
    SB_SCR_PORT_STREAM(span * 8, 16 * span, 8 * span, N / span / 2, P_rotate_R);
    SB_SCR_WRITE(P_rotate_A, N * 4, 0);
    SB_SCR_WRITE(P_rotate_B, N * 4, N * 4);

    //for (int j = 0; j < blocks; ++j) {
      //SB_DMA_READ(w, blocks * 8, 8, span, P_rotate_W);
    //}
    SB_CONST(P_rotate_W, 1065353216, 1);
    SB_CONST(P_rotate_Rotate, *((uint64_t*)(w + blocks)), N / 2 - 1);
    SB_CONST(P_rotate_Current, 1065353216, 1);
    SB_RECURRENCE(P_rotate_Angle, P_rotate_W, N / 2 - 1);
    SB_RECURRENCE(P_rotate_Rec, P_rotate_Current, N / 2 - 2);
    SB_GARBAGE(P_rotate_Rec, 1);

    SB_WAIT_SCR_WR();
    //SB_WAIT_ALL();
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
  SB_WAIT_ALL();

  SB_CONFIG(subadd_config, subadd_size);
  SB_SCR_PORT_STREAM(0       , 16 * span, 8 * span, N / span / 2, P_subadd_L);
  SB_SCR_PORT_STREAM(span * 8, 16 * span, 8 * span, N / span / 2, P_subadd_R);
  SB_DMA_WRITE(P_subadd_A, 16 * span, 8 * span, N / span / 2, a);
  SB_DMA_WRITE(P_subadd_B, 16 * span, 8 * span, N / span / 2, a + span);
  SB_WAIT_ALL();

}
