#include <complex>
#include <cmath>
#include <algorithm>
#include "sb_insts.h"
#include "sim_timing.h"
#include "compute0.h"
#include "compute1.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;
complex<float> _buffer[N];
#define PI 3.14159265358979303

void fft(complex<float> *_a, complex<float> *w) {
  SB_CONTEXT(1 | 2 | 4 | 8);
  SB_CONFIG(compute0_config, compute0_size);

  complex<float> *from = _a, *to = _buffer;
  int span = 2;
  int blocks;

  SB_CONTEXT(1);
  for (blocks = N / 2; span < 8; blocks >>= 1, span <<= 1) {
    SB_DMA_READ(from,          2 * blocks * 8, blocks * 8, span / 2, P_compute0_L);
    SB_DMA_READ(from + blocks, 2 * blocks * 8, blocks * 8, span / 2, P_compute0_R);

    SB_REPEAT_PORT(blocks / 2);
    SB_DMA_READ(w, 8 * blocks, 8, span / 2, P_compute0_W);

    SB_DMA_WRITE(P_compute0_A, 8, 8, N / 2, to);
    SB_DMA_WRITE(P_compute0_B, 8, 8, N / 2, to + N / 2);

    SB_WAIT_ALL();
    swap(from, to);
  }

  for ( ; blocks != 1; blocks >>= 1, span <<= 1) {
    int _span = span / 8;
    for (int offset = 0, ctx = 1; offset < span / 2; offset += _span, ctx <<= 1) {
      //printf("%d\n", ctx);
      SB_CONTEXT(ctx);

      SB_DMA_READ(from + (2 * blocks) * offset         , 2 * blocks * 8, blocks * 8, _span, P_compute0_L);
      SB_DMA_READ(from + (2 * blocks) * offset + blocks, 2 * blocks * 8, blocks * 8, _span, P_compute0_R);

      SB_REPEAT_PORT(blocks / 2);
      SB_DMA_READ(w + blocks * offset, 8 * blocks, 8, _span, P_compute0_W);

      SB_DMA_WRITE(P_compute0_A, 8, 8, N / 8, to +         offset);
      SB_DMA_WRITE(P_compute0_B, 8, 8, N / 8, to + N / 2 + offset);
    }

    SB_WAIT_ALL();
    swap(from, to);
  }

  SB_CONFIG(compute1_config, compute1_size);
  SB_DMA_READ(from,     16, 8, N / 2, P_compute1_L)
  SB_DMA_READ(from + 1, 16, 8, N / 2, P_compute1_R)
  SB_DMA_READ(w, 8, 8, N / 2, P_compute1_W);
  SB_DMA_WRITE(P_compute1_A, 8, 8, N / 2, to);
  SB_DMA_WRITE(P_compute1_B, 8, 8, N / 2, to + N / 2);
  SB_WAIT_ALL();
  //swap(from, to);

  if (to != _a) {
    for (int i = 0; i < N; ++i) {
      _a[i] = _buffer[i];
    }
  }
}
