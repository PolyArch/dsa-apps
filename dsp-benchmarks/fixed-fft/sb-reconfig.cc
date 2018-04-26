#include <complex>
#include <cmath>
#include <algorithm>
#include "sb_insts.h"
#include "compute0.dfg.h"
#include "compute1.dfg.h"
#include "sim_timing.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

static bool _first_time = true;

complex<float> *fft(complex<float> *from, complex<float> *to, complex<float> *w) {
  if (!_first_time)
    begin_roi();

  SB_CONFIG(compute0_config, compute0_size);

  int blocks = _N_ / 2;
  int span = _N_ / blocks;
  complex<float> *_w = w + _N_ - 2;
  for ( ; blocks != 1; blocks >>= 1, span <<= 1) {
    _w -= span / 2;
    SB_DMA_READ(from,          2 * blocks * 8, blocks * 8, span / 2, P_compute0_L);
    SB_DMA_READ(from + blocks, 2 * blocks * 8, blocks * 8, span / 2, P_compute0_R);

    SB_REPEAT_PORT(blocks / 4);
    SB_DMA_READ(_w, 0, 4 * span, 1, P_compute0_W);

    SB_DMA_WRITE(P_compute0_A, 8, 8, _N_ / 2, to);
    SB_DMA_WRITE(P_compute0_B, 8, 8, _N_ / 2, to + _N_ / 2);

    SB_WAIT_ALL();
    swap(from, to);
  }

  SB_CONFIG(compute1_config, compute1_size);
  SB_DMA_READ(from,     16, 8, _N_ / 2, P_compute1_L)
  SB_DMA_READ(from + 1, 16, 8, _N_ / 2, P_compute1_R)
  SB_DMA_READ(w, 8, 8, _N_ / 2, P_compute1_W);
  SB_DMA_WRITE(P_compute1_A, 8, 8, _N_ / 2, to);
  SB_DMA_WRITE(P_compute1_B, 8, 8, _N_ / 2, to + _N_ / 2);
  SB_WAIT_ALL();
  //swap(from, to);

  if (!_first_time)
    end_roi();
  else
    _first_time = false;

  return to;
}
