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

void fft(complex<float> *_a, complex<float> *_w) {
  int from = 0, to = 16384, w = 32768;
  SB_DMA_SCRATCH_LOAD(_a, 8, 8, N, 0);
  //SB_DMA_SCRATCH_LOAD(_w, 8, 8, N / 2, w);
  SB_CONFIG(compute0_config, compute0_size);

  int span = 2;
  for (int blocks = N / 2; blocks != 1; blocks >>= 1, span <<= 1) {
    SB_WAIT_SCR_WR();

    SB_SCR_PORT_STREAM(from,              2 * blocks * 8, blocks * 8, span / 2, P_compute0_L);
    SB_SCR_PORT_STREAM(from + blocks * 8, 2 * blocks * 8, blocks * 8, span / 2, P_compute0_R);

    SB_REPEAT_PORT(blocks / 2);
    SB_DMA_READ(_w, 8 * blocks, 8, span / 2, P_compute0_W);

    SB_SCR_WRITE(P_compute0_A, 8 * N / 2, to);
    SB_SCR_WRITE(P_compute0_B, 8 * N / 2, to + N / 2 * 8);

    //swap(from, to);
    from ^= to;
    to ^= from;
    from ^= to;
  }

  SB_WAIT_SCR_WR();
  SB_CONFIG(compute1_config, compute1_size);
  SB_SCR_PORT_STREAM(from,     16, 8, N / 2, P_compute1_L)
  SB_SCR_PORT_STREAM(from + 8, 16, 8, N / 2, P_compute1_R)
  SB_DMA_READ(_w, 8, 8, N / 2, P_compute1_W);
  SB_DMA_WRITE(P_compute1_A, 8, 8, N / 2, _a);
  SB_DMA_WRITE(P_compute1_B, 8, 8, N / 2, _a + N / 2);
  SB_WAIT_ALL();
}
