#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "multi.dfg.h"
#include "sb_insts.h"

using std::complex;

void cholesky(complex<float> *a, complex<float> *L) {
  SB_CONFIG(multi_config, multi_size);
  {
    SB_CONST(P_multi_VAL, *((uint64_t *) a), 1);
    SB_REPEAT_PORT(_N_ - 1);
    SB_RECURRENCE(P_multi_invsqrt, P_multi_DIV, 1);
    SB_REPEAT_PORT((_N_ - 1) * _N_ / 2);
    SB_RECURRENCE(P_multi_invpure, P_multi_V, 1);
    SB_DMA_READ(a + 1, 0, 8 * (_N_ - 1), 1, P_multi_VEC);
    SB_DMA_WRITE(P_multi_fin, 8 * _N_, 8, _N_ - 1, L + _N_);
    SB_DMA_WRITE(P_multi_sqrt, 0, 8, 1, L);
    SB_DMA_READ_STRETCH(a + _N_ + 1, 8 * (_N_ + 1), 8 * (_N_ - 1), -8, _N_ - 1, P_multi_Z);
    SB_CONFIG_PORT(_N_ - 1, -1);
    SB_DMA_READ(a + 1, 8, 8, _N_ - 1, P_multi_A);
    SB_DMA_READ_STRETCH(a + 1, 8, 8 * (_N_ - 1), -8, _N_ - 1, P_multi_B);
  }
  int addr = 0;
  int array = 512;
  for (int i = 1; i < _N_; ++i) {
    int total = _N_ - i - 1;
    SB_RECURRENCE(P_multi_O, P_multi_VAL, 1);
    SB_SCR_WRITE(P_multi_O, total * 8, addr);

    //SB_RECURRENCE(P_multi_O, P_multi_Z, total * (_N_ - i) / 2);
    SB_SCR_WRITE(P_multi_O, total * (_N_ - i) * 4, array);
    SB_WAIT_SCR_WR();
    SB_SCRATCH_READ(array, total * (_N_ - i) * 4, P_multi_Z);

    SB_DMA_WRITE(P_multi_sqrt, 0, 8, 1, L + i * _N_ + i);
    SB_REPEAT_PORT(total);
    SB_RECURRENCE(P_multi_invsqrt, P_multi_DIV, 1);
    SB_REPEAT_PORT((1 + total) * total / 2);
    SB_RECURRENCE(P_multi_invpure, P_multi_V, 1);
    SB_WAIT_SCR_WR();
    SB_SCR_PORT_STREAM(addr, 8, 8, total, P_multi_VEC);
    SB_DMA_WRITE(P_multi_fin, 8 * _N_, 8, total, L + (i + 1) * _N_ + i);
    SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * total, -8, total, P_multi_B);
    SB_CONFIG_PORT(total, -1);
    SB_SCR_PORT_STREAM(addr, 8, 8, total, P_multi_A);
    addr ^= 256;
  }
  SB_WAIT_ALL();
}

