#include <complex>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "sb_insts.h"
#include "sim_timing.h"
#include "compute.dfg.h"
#include "fine0.dfg.h"
#include "fine1.dfg.h"
#include "fine2.dfg.h"

using std::complex;

static bool _first_time = true;

complex<int16_t> *fft(complex<int16_t> *from, complex<int16_t> *to, complex<float> *w) {
  if (!_first_time)
    begin_roi();

  int N = _N_;
  complex<float> *_w = w + _N_ - 2;

  SB_CONFIG(compute_config, compute_size);

  int blocks = N / 2;
  int span = N / blocks;

  SB_DMA_READ(from,          blocks * 8, blocks * 4, span / 2, P_compute_L);
  SB_DMA_READ(from + blocks, blocks * 8, blocks * 4, span / 2, P_compute_R);
  SB_CONST(P_compute_W, 1065353216, N / 16);

  SB_SCR_WRITE(P_compute_A, N * 2, 0);
  SB_SCR_WRITE(P_compute_B, N * 2, N * 2);
  SB_WAIT_SCR_WR();

  blocks >>= 1;
  span <<= 1;

  int scr = 0;

  while (blocks >= 8) {
    _w -= span / 2;

    SB_CONTEXT(1);
    SB_SCR_PORT_STREAM(scr             , blocks * 8, blocks * 4, span / 2, P_compute_L);
    SB_SCR_PORT_STREAM(scr + blocks * 8, blocks * 8, blocks * 4, span / 2, P_compute_R);
    SB_REPEAT_PORT(blocks / 8);
    SB_DMA_READ(_w, 0, 4 * span, 1, P_compute_W);

    scr ^= 4096;
    SB_SCR_WRITE(P_compute_A, N * 2, scr);
    SB_SCR_WRITE(P_compute_B, N * 2, scr + N * 2);
    SB_WAIT_SCR_WR();

    blocks >>= 1;
    span <<= 1;
  }

  _w -= span / 2;
  SB_WAIT_ALL();
  
  SB_CONFIG(fine2_config, fine2_size);
  SB_SCRATCH_READ(scr, 4 * N, P_fine2_V);
  SB_DMA_READ(_w, 8, 8, N / 8, P_fine2_W);
  scr ^= 4096;
  SB_SCR_WRITE(P_fine2_A, N * 2, scr);
  SB_SCR_WRITE(P_fine2_B, N * 2, scr + N * 2);
  _w -= span / 4;
  SB_WAIT_ALL();

  SB_CONFIG(fine1_config, fine1_size);
  SB_DMA_READ(_w, 8, 8, N / 4, P_fine1_W);
  SB_SCRATCH_READ(scr, N * 4, P_fine1_V);
  scr ^= 4096;
  SB_SCR_WRITE(P_fine1_A, N * 2, scr);
  SB_SCR_WRITE(P_fine1_B, N * 2, scr + N * 2);
  SB_WAIT_ALL();

  SB_CONFIG(fine0_config, fine0_size);
  SB_DMA_READ(w, 8, 8, N / 2, P_fine0_W);
  SB_SCRATCH_READ(scr, N * 4, P_fine0_V);
  SB_DMA_WRITE(P_fine0_A, 0, 2 * N, 1, to);
  SB_DMA_WRITE(P_fine0_B, 0, 2 * N, 1, to + N / 2);
  SB_WAIT_ALL();

  if (!_first_time)
    end_roi();
  else
    _first_time = false;

  return to;
}