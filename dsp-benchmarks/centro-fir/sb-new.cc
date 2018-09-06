#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "ss-config/fixed_point.h"
#include "sb_insts.h"

#include "compute.dfg.h"
#define get_pad(n, vec_width) ((n) % (vec_width) ? (vec_width) - ((n) % vec_width) : 0)

using std::complex;

#define VEC 4

complex<float> _zero(0, 0);

void filter(int n, int m, complex<float> *a, complex<float> *b, complex<float> *c, complex<float> *) {
  SB_CONFIG(compute_config, compute_size);
  SB_DMA_SCRATCH_LOAD(a, 8, 8, n, 0);
  SB_WAIT_SCR_WR();
  int half = m / 2;
  const int block = 128;
  for (int io = 0; io < n - m + 1; io += block) {
    int len = std::min(n - m + 1 - io, block);
    int pad = get_pad(len, VEC);
    SB_CONST(P_compute_C, 0, len + pad);
    SB_RECURRENCE(P_compute_O, P_compute_C, half * (len + pad));
    SB_REPEAT_PORT((len + pad) / VEC);
    SB_DMA_READ(b, 0, 8 * half, 1, P_compute_B);
    SB_SCR_PORT_STREAM(io * 8, 8, 8 * (len + pad), half, P_compute_AL);
    //This crashes!
    SB_SCR_PORT_STREAM((io + m - 1) * 8, -8, 8 * (len + pad), half, P_compute_AR);
    //for (int j = 0; j < half; ++j) {
    //  SB_SCRATCH_READ((io + m - 1 - j) * 8, 8 * (len + pad), P_compute_AR);
    //}
    SB_SCRATCH_READ((half + io) * 8, 8 * len, P_compute_AL);
    SB_CONST(P_compute_AL, 0, pad);
    SB_CONST(P_compute_AR, 0, len + pad);
    SB_CONST(P_compute_B, *((uint64_t*) b + half), (len + pad) / VEC);
    SB_DMA_WRITE(P_compute_O, 0, 8 * (len + pad), 1, c + io);
    //for (int ii = 0; ii < len; ++ii) {
    //  int i = io + ii;
    //  for (int j = 0, jj = m - 1; j < m / 2; ++j, --jj) {
    //    c[i] += (a[i + j] + a[i + jj]) * b[j];
    //  }
    //  c[i] += a[i + m / 2] * b[m / 2];
    //}
  }
  SB_WAIT_ALL();
}
