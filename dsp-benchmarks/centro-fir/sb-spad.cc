#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "ss-config/fixed_point.h"
#include "sb_insts.h"

#include "compute.h"

#define complexmidul(a, b) \
  (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conjmidul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

#define PI 3.14159265358979303

#define VEC 4

complex<float> _zero(0, 0);

void filter(int n, int m, complex<float> *a, complex<float> *b, complex<float> *c, complex<float> *) {
  SB_CONFIG(compute_config, compute_size);
  int len = n - m + 1;
  int pad = (VEC - (len & (VEC - 1))) & (VEC - 1);
  int mid = m >> 1;
  SB_DMA_SCRATCH_LOAD(a, 8, 8, n, 0);
  SB_WAIT_ALL();
  SB_REPEAT_PORT((len + pad) / VEC);
  SB_DMA_READ(b, 8, 8, mid, P_compute_B);
  SB_CONST(P_compute_C, 0, len + pad);
  SB_RECURRENCE(P_compute_O, P_compute_C, mid * (len + pad));
  for (int i = 0; i < mid; ++i) {
    SB_SCR_PORT_STREAM(i * 8          ,  0, 8 * len, 1, P_compute_AL);
    SB_CONST(P_compute_AL, 0, pad);
    SB_SCR_PORT_STREAM((m - 1 - i) * 8, 0, 8 * len, 1, P_compute_AR);
    SB_CONST(P_compute_AR, 0, pad);
  }
  /*for (int j = 0; j < mid; ++j) {
    for (int i = 0; i < len; ++i) {
      complex<float> delta0(complex_mul(a[i + j], b[j]));
      complex<float> delta1(complex_mul(a[i + (m - j - 1)], b[j]));
      complex<float> delta(complex_add(delta0, delta1));
      c[i] = complex<float>(complex_add(c[i], delta));
    }
  }*/
  SB_SCR_PORT_STREAM(mid * 8, 0, 8 * len, 1, P_compute_AL);
  SB_CONST(P_compute_AL, 0, pad);
  SB_CONST(P_compute_AR, *((uint64_t*)&_zero), len + pad);
  SB_CONST(P_compute_B, *((uint64_t*) b + mid), (len + pad) / VEC);
  SB_DMA_WRITE(P_compute_O, 8, 8, len + pad, c);
  SB_WAIT_ALL();
  /*for (int i = 0; i < len; ++i) {
    complex<float> delta(complex_mul(a[i + mid + 1], b[mid + 1]));
    c[i] = complex<float>(complex_add(c[i], delta));
  }*/
}
