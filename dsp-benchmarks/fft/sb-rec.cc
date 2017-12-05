#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "sb_insts.h"
#include "sim_timing.h"
#include "compute_rec.h"
#include "compute1.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;
static complex<float> buffer[N];
#define PI 3.14159265358979303

void fft(complex<float> *a, complex<float> *_w) {
  SB_CONFIG(compute_rec_config, compute_rec_size);

  SB_DMA_READ(a        , 0, N * 4, 1, P_compute_rec_L);
  SB_DMA_READ(a + N / 2, 0, N * 4, 1, P_compute_rec_R);
  SB_CONST(P_compute_rec_W, *((uint64_t*) _w), N / 4);
  SB_2D_CONST(P_compute_rec_MUX, 0, N / 8, 1, N / 8, 1);
/*
  SB_DMA_WRITE(P_compute_rec_AEVEN, 8, 8, N / 4, buffer);
  SB_DMA_WRITE(P_compute_rec_AODD, 8, 8, N / 4, buffer + N / 4);
  SB_DMA_WRITE(P_compute_rec_BEVEN, 8, 8, N / 4, buffer + N / 2);
  SB_DMA_WRITE(P_compute_rec_BODD, 8, 8, N / 4, buffer + N / 4 * 3);

  SB_WAIT_ALL();

  for (int i = 0; i < N; ++i)
    std::cout << buffer[i] << "\n";

  return ;
*/
  SB_SCR_WRITE(P_compute_rec_BEVEN, N * 2, 0);
  SB_SCR_WRITE(P_compute_rec_BODD,  N * 2, N * 2);

  int span = 4;
  int blocks = N / span;
  int cnt = 0;
  for (; blocks > 2; blocks >>= 1, span <<= 1, ++cnt) {

    SB_RECURRENCE(P_compute_rec_AEVEN, P_compute_rec_L, N / 4);
    SB_RECURRENCE(P_compute_rec_AODD,  P_compute_rec_R, N / 4);
    SB_WAIT_SCR_WR();

    //SB_RECURRENCE(P_compute_rec_BEVEN, P_compute_rec_L, N / 4);
    //SB_RECURRENCE(P_compute_rec_BODD,  P_compute_rec_R, N / 4);

    SB_SCRATCH_READ(0    , N * 2, P_compute_rec_L);
    SB_SCRATCH_READ(N * 2, N * 2, P_compute_rec_R);

    SB_REPEAT_PORT(blocks / 2);
    SB_DMA_READ(_w, 8 * blocks, 8, span / 2, P_compute_rec_W);

    SB_2D_CONST(P_compute_rec_MUX, 0, blocks / 4, 1, blocks / 4, span / 2);

/*if (blocks == 4) {

  SB_DMA_WRITE(P_compute_rec_AEVEN, blocks * 8, blocks * 4, span / 2, buffer);
  SB_DMA_WRITE(P_compute_rec_AODD, blocks * 8, blocks * 4, span / 2, buffer + blocks / 2);
  SB_DMA_WRITE(P_compute_rec_BEVEN, blocks * 8, blocks * 4, span / 2, buffer + N / 2);
  SB_DMA_WRITE(P_compute_rec_BODD, blocks * 8, blocks * 4, span / 2, buffer + N / 2 + blocks / 2);

  SB_WAIT_ALL();

  for (int i = 0; i < N; ++i)
    std::cout << buffer[i] << "\n";

  return ;
}*/

    SB_SCR_WRITE(P_compute_rec_BEVEN, N * 2, 0);
    SB_SCR_WRITE(P_compute_rec_BODD,  N * 2, N * 2);

  }

  SB_RECURRENCE(P_compute_rec_AEVEN, P_compute_rec_L, N / 4);
  SB_RECURRENCE(P_compute_rec_AODD,  P_compute_rec_R, N / 4);
  SB_SCRATCH_READ(0    , N * 2, P_compute_rec_L);
  SB_SCRATCH_READ(N * 2, N * 2, P_compute_rec_R);
  SB_DMA_READ(_w, 16, 8, N / 4, P_compute_rec_W);
  SB_CONST(P_compute_rec_MUX, 0, N / 4);

  SB_DMA_WRITE(P_compute_rec_AEVEN, 8, 8, N / 2, a);
  SB_DMA_WRITE(P_compute_rec_BEVEN, 8, 8, N / 2, a + N / 2);
  SB_WAIT_ALL();

  //for (int i = 0; i < N; ++i)
    //std::cout << a[i] << "\n";

  SB_CONFIG(compute1_config, compute1_size);
  SB_DMA_READ(a    , 16, 8, N / 2, P_compute1_L);
  SB_DMA_READ(a + 1, 16, 8, N / 2, P_compute1_R);
  //SB_SCRATCH_READ(0    , N * 4, P_compute1_L);
  //SB_SCRATCH_READ(N * 4, N * 4, P_compute1_R);
  SB_DMA_READ(_w, 8, 8, N / 2, P_compute1_W);
  SB_DMA_WRITE(P_compute1_A, 8, 8, N / 2, a);
  SB_DMA_WRITE(P_compute1_B, 8, 8, N / 2, a + N / 2);
  SB_WAIT_ALL();
}
