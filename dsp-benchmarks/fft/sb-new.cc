#include <complex>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "sb_insts.h"
#include "sim_timing.h"
#include "compute0.dfg.h"
#include "compute1.dfg.h"

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

complex<float> *fft(complex<float> *from, complex<float> *to, complex<float> *w) {
  SB_CONFIG(compute0_config, compute0_size);

  int blocks = N / 2;
  int span = N / blocks;
  for ( ; blocks != 1; blocks >>= 1, span <<= 1) {
    SB_DMA_READ(from,          2 * blocks * 8, blocks * 8, span / 2, P_compute0_L);
    SB_DMA_READ(from + blocks, 2 * blocks * 8, blocks * 8, span / 2, P_compute0_R);

    SB_REPEAT_PORT(blocks / 2);
    SB_DMA_READ(w, 8 * blocks, 8, span / 2, P_compute0_W);

    SB_DMA_WRITE(P_compute0_A, 8, 8, N / 2, to);
    SB_DMA_WRITE(P_compute0_B, 8, 8, N / 2, to + N / 2);

    swap(from, to);
    SB_WAIT_ALL();
  }

  SB_CONFIG(compute1_config, compute1_size);
  SB_DMA_READ(from,     16, 8, N / 2, P_compute1_L)
  SB_DMA_READ(from + 1, 16, 8, N / 2, P_compute1_R)
  SB_DMA_READ(w, 8, 8, N / 2, P_compute1_W);
  SB_DMA_WRITE(P_compute1_A, 8, 8, N / 2, to);
  SB_DMA_WRITE(P_compute1_B, 8, 8, N / 2, to + N / 2);
  SB_WAIT_ALL();

  return to;
}
