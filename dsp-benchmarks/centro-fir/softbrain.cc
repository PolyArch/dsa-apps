#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "softbrain-config/fixed_point.h"
#include "sb_insts.h"

#include "compute.h"

#define complex_mul(a, b) \
  (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

#define PI 3.14159265358979303

complex<float> _zero(0, 0);

void filter(int n, int m, complex<float> *a, complex<float> *b, complex<float> *c) {
  SB_CONFIG(compute_config, compute_size);
  int len = n - m + 1;
  int pad = len & 1;
  int _m = (m >> 1) - 1;
  SB_DMA_SCRATCH_LOAD(a, 8, 8, n, _m * 8);
  SB_WAIT_ALL();
  SB_SCR_PORT_STREAM(m * 4              ,  8, 8 * len, _m, P_compute_AL); SB_CONST(P_compute_AL, 0, pad);
  SB_SCR_PORT_STREAM(m * 4 + (m - 1) * 8, -8, 8 * len, _m, P_compute_AR); SB_CONST(P_compute_AR, 0, pad);
  //SB_REPEAT_PORT(len);
  SB_REPEAT_PORT((len + pad) / 2);
  SB_DMA_READ(b, 8, 8, _m, P_compute_B);
  SB_CONST(P_compute_C, 0, len + pad);
  SB_RECURRENCE(P_compute_O, P_compute_C, _m * (len + pad));
  /*for (int j = 0; j < _m; ++j) {
    for (int i = 0; i < len; ++i) {
      complex<float> delta0(complex_mul(a[i + j], b[j]));
      complex<float> delta1(complex_mul(a[i + (m - j - 1)], b[j]));
      complex<float> delta(complex_add(delta0, delta1));
      c[i] = complex<float>(complex_add(c[i], delta));
    }
  }*/
  SB_DMA_READ(a + _m + 1, 0, 8 * len, 1, P_compute_AL); SB_CONST(P_compute_AL, 0, pad);
  SB_CONST(P_compute_AR, *((uint64_t*)&_zero), len + pad);
  SB_CONST(P_compute_B, *((uint64_t*) c + _m + 1), (len + pad) / 2);
  SB_DMA_WRITE(P_compute_O, 8, 8, len, c);
  SB_GARBAGE(P_compute_O, pad);
  SB_WAIT_ALL();
  /*for (int i = 0; i < len; ++i) {
    complex<float> delta(complex_add(a[i + _m + 1], b[_m + 1]));
    c[i] = complex<float>(complex_add(c[i], delta));
  }*/
}
