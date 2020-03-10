#include <complex>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "ss_insts.h"
#include "sim_timing.h"
#include "temporal0.dfg.h"
#include "temporal1.dfg.h"
#include "temporal2.dfg.h"

using std::complex;

static bool _first_time = true;

complex<float> *fft(complex<float> *from, complex<float> *to, complex<float> *w) {
  //if (!_first_time)
  //  begin_roi();

  int N = _N_;
  complex<float> *_w = w + _N_ - 2;

  SS_CONFIG(temporal0_config, temporal0_size);

  int blocks = N / 2;
  int span = N / blocks;

  //SS_DMA_READ(from,          2 * blocks * 8, blocks * 8, span / 2, P_temporal0_L);
  //SS_DMA_READ(from + blocks, 2 * blocks * 8, blocks * 8, span / 2, P_temporal0_R);
  //SS_CONST(P_temporal0_W, 1065353216, N / 8);
    
  //SS_SCR_WRITE(P_temporal0_A, N * 4, 0);
  //SS_SCR_WRITE(P_temporal0_B, N * 4, N * 4);
  //SS_WAIT_SCR_WR();
    
  //blocks >>= 1;
  //span <<= 1;

  int scr = 0;

  while (blocks >= 4) {

    //SS_CONTEXT(1);
    SS_SCR_PORT_STREAM(scr             , 2 * blocks * 8, blocks * 8, span / 2, P_temporal0_L);
    SS_SCR_PORT_STREAM(scr + blocks * 8, 2 * blocks * 8, blocks * 8, span / 2, P_temporal0_RR);
    SS_REPEAT_PORT(blocks / 4);
    SS_DMA_READ(_w, 0, 4 * span, 1, P_temporal0_W);

    scr ^= 8192;
    SS_SCR_WRITE(P_temporal0_A, N * 4, scr);
    SS_SCR_WRITE(P_temporal0_B, N * 4, scr + N * 4);
    SS_WAIT_SCR_WR();

    blocks >>= 1;
    span <<= 1;
    _w -= span / 2;
  }

  SS_WAIT_ALL();
  
  SS_CONFIG(temporal1_config, temporal1_size);
  SS_SCRATCH_READ(scr, 8 * N, P_temporal1_V);
  SS_DMA_READ(_w, 0, N * 2, 1, P_temporal1_W);
  scr ^= 8192;
  SS_SCR_WRITE(P_temporal1_A, N * 4, scr);
  SS_SCR_WRITE(P_temporal1_B, N * 4, scr + N * 4);
  SS_WAIT_ALL();

  SS_CONFIG(temporal2_config, temporal2_size);
  SS_DMA_READ(w, 8, 8, N / 2, P_temporal2_W);
  SS_SCRATCH_READ(scr, N * 8, P_temporal2_V);
  SS_DMA_WRITE(P_temporal2_A, 0, 4 * N, 1, to);
  SS_DMA_WRITE(P_temporal2_B, 0, 4 * N, 1, to + N / 2);
  SS_WAIT_ALL();

  //if (!_first_time)
  //  end_roi();
  //else
  //  _first_time = false;

  return to;
}
