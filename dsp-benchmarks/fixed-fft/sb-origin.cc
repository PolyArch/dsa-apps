#include <complex>
#include <cmath>
#include <algorithm>
#include "sb_insts.h"
#include "compute0.dfg.h"
#include "compute1.dfg.h"
#include "sim_timing.h"
#include "fine0.dfg.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

static bool _first_time = true;

complex<int16_t> *fft(complex<int16_t> *from, complex<int16_t> *to, complex<float> *w) {
  if (!_first_time)
    begin_roi();

  SB_CONFIG(compute0_config, compute0_size);

  int blocks = _N_ / 2;
  int span = _N_ / blocks;
  for ( ; blocks != 2; blocks >>= 1, span <<= 1) {
    SB_DMA_READ(from,          blocks * 8, blocks * 4, span / 2, P_compute0_L);
    SB_DMA_READ(from + blocks, blocks * 8, blocks * 4, span / 2, P_compute0_R);

    for (int j = 0; j < span / 2; ++j) {
      SB_CONST(P_compute0_W, *((unsigned long long*)(w + j * blocks)), blocks / 4);
      /*for (int i = 0; i < blocks; ++i) {
        //printf("%d %d %d\n", blocks, j, i);
        complex<float> &L = from[2 * j + i];
        complex<float> &R = from[2 * j + i + blocks];
        complex<float> tmp(complex_mul(w[j], R));
        to[i + j] = complex<float>(complex_add(L, tmp));
        to[i + j + span / 2 * blocks] = complex<float>(complex_sub(L, tmp));
      }*/
    }

    SB_DMA_WRITE(P_compute0_A, 8, 8, _N_ / 4, to);
    SB_DMA_WRITE(P_compute0_B, 8, 8, _N_ / 4, to + _N_ / 2);

    SB_WAIT_ALL();
    swap(from, to);
  }

  SB_CONFIG(compute1_config, compute1_size);
  SB_DMA_READ(from,     16, 8, _N_ / 4, P_compute1_L)
  SB_DMA_READ(from + 1, 16, 8, _N_ / 4, P_compute1_R)
  SB_DMA_READ(w, 16, 8, _N_ / 4, P_compute1_W);
  SB_DMA_WRITE(P_compute1_A, 8, 8, _N_ / 4, to);
  SB_DMA_WRITE(P_compute1_B, 8, 8, _N_ / 4, to + _N_ / 2);
  SB_WAIT_ALL();

  SB_CONFIG(fine0_config, fine0_size);
  SB_DMA_READ(to, 8, 8, _N_ / 2, P_fine0_V);
  SB_DMA_READ(w, 8, 8, _N_ / 2, P_fine0_W);
  SB_DMA_WRITE(P_fine0_A, 8, 8, _N_ / 4, from);
  SB_DMA_WRITE(P_fine0_B, 8, 8, _N_ / 4, from + _N_ / 2);
  SB_WAIT_ALL();
  //swap(from, to);

  if (!_first_time)
    end_roi();
  else
    _first_time = false;

  return from;
}
