#include <complex>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "sb_insts.h"
#include "sim_timing.h"
#include "compute.dfg.h"
#include "fine.dfg.h"

using std::complex;

static bool _first_time = true;

complex<float> *fft(complex<float> *from, complex<float> *to, complex<float> *w) {
  //if (!_first_time)
    //begin_roi();

  int N = _N_;
  complex<float> *_w = w + _N_ - 2;

  SB_CONFIG(compute_config, compute_size);

  int blocks = N / 2;
  int span = N / blocks;

  SB_DMA_READ(from,          2 * blocks * 8, blocks * 8, span / 2, P_compute_L);
  SB_DMA_READ(from + blocks, 2 * blocks * 8, blocks * 8, span / 2, P_compute_R);
  SB_CONST(P_compute_W, 1065353216, N / 8);

  SB_SCR_WRITE(P_compute_A, N * 4, 0);
  SB_SCR_WRITE(P_compute_B, N * 4, N * 4);
  SB_WAIT_SCR_WR();

  blocks >>= 1;
  span <<= 1;

  int scr = 0;

  while (blocks >= 4) {
    _w -= span / 2;

    //SB_CONTEXT(1);
    SB_SCR_PORT_STREAM(scr             , 2 * blocks * 8, blocks * 8, span / 2, P_compute_L);
    SB_SCR_PORT_STREAM(scr + blocks * 8, 2 * blocks * 8, blocks * 8, span / 2, P_compute_R);
    SB_REPEAT_PORT(blocks / 4);
    SB_DMA_READ(_w, 0, 4 * span, 1, P_compute_W);

    blocks >>= 1;
    span <<= 1;
    scr ^= 8192;
    SB_SCR_WRITE(P_compute_A, N * 4, scr);
    SB_SCR_WRITE(P_compute_B, N * 4, scr + N * 4);
    SB_WAIT_SCR_WR();

  }

  _w -= span / 2;
  SB_WAIT_ALL();
  
  SB_CONFIG(fine_config, fine_size);
  SB_DMA_READ(_w, 0, N * 4, 1, P_fine_W);
  SB_SCRATCH_READ(scr, 8 * N, P_fine_V);
  scr ^= 8192;
  SB_RECURRENCE(P_fine_A, P_fine_V_, N / 2)
  SB_SCR_WRITE(P_fine_B, N * 4, scr + N * 4);
  SB_DMA_READ(w, 0, N * 2, 1, P_fine_W_);
  SB_DMA_WRITE(P_fine_A_, 0, 4 * N, 1, to);
  SB_DMA_WRITE(P_fine_B_, 0, 4 * N, 1, to + N / 2);
  SB_WAIT_SCR_WR();
  SB_SCRATCH_READ(scr + N * 4, N * 4, P_fine_V_);
  SB_WAIT_ALL();

  //if (!_first_time)
    //end_roi();
  //else
    //_first_time = false;

  return to;
}
