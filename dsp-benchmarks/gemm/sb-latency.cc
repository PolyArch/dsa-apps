#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include "ss-config/fixed_point.h"
#include "compute.dfg.h"
#include "sb_insts.h"

#define complex_mul(a, b) \
  FIX_MINUS(FIX_MUL((a).real(), (b).real()), FIX_MUL((a).imag(), (b).imag())), \
  FIX_ADD(FIX_MUL((a).real(), (b).imag()), FIX_MUL((a).imag(), (b).real()))

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) FIX_ADD((a).real(), (b).real()), FIX_ADD((a).imag(), (b).imag())

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

#define PI 3.14159265358979303

void gemm(int n, int m, int p, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  SB_CONTEXT(255);
  SB_CONFIG(compute_config, compute_size);
  SB_DMA_SCRATCH_LOAD(b, 0, 8 * m * p / 2, 1, 0);
  SB_WAIT_SCR_WR();
  int resudo = n & 7;
  int _n = n - resudo;
  int io;
  for (io = 0; io < _n; io += 8) {
    SB_CONTEXT_I(255);
    SB_CONST(P_compute_C, 0, p / 2);
    SB_RECURRENCE(P_compute_O, P_compute_C, (p / 2) * (m / 2 - 1));
    SB_SCR_PORT_STREAM(0    , 8 * p, 4 * p, m / 2, P_compute_BE);
    SB_SCR_PORT_STREAM(p * 4, 8 * p, 4 * p, m / 2, P_compute_BO);

      SB_STRIDE(8, 8);

      complex<int16_t> *_c = c + io * p;
      complex<int16_t> *_a = a + io * m;

      SB_CONTEXT_OFFSET(255, p);
      SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      SB_CONTEXT_OFFSET(255, m)
      SB_REPEAT_PORT(p / 4);
      SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      SB_CONTEXT_OFFSET(255, 0);

      //SB_CONTEXT_I(1);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      //_c += p;
      //_a += m;

      //SB_CONTEXT_I(2);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      //_c += p;
      //_a += m;

      //SB_CONTEXT_I(4);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      //_c += p;
      //_a += m;

      //SB_CONTEXT_I(8);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      //_c += p;
      //_a += m;

      //SB_CONTEXT_I(16);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      //_c += p;
      //_a += m;

      //SB_CONTEXT_I(32);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      //_c += p;
      //_a += m;

      //SB_CONTEXT_I(64);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      //_c += p;
      //_a += m;

      //SB_CONTEXT_I(128);
      //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      //SB_REPEAT_PORT(p / 4);
      //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);

    /*for (int k = 0; k < m; k += 2) {
      SB_DMA_READ(a + i * m + k, 0, 8, p / 4, P_compute_A);
    }*/
  }

  if (resudo) {

    SB_CONTEXT_I(15);
    SB_CONST(P_compute_C, 0, p / 2);
    SB_RECURRENCE(P_compute_O, P_compute_C, (p / 2) * (m / 2 - 1));
    SB_SCR_PORT_STREAM(0    , 8 * p, 4 * p, m / 2, P_compute_BE);
    SB_SCR_PORT_STREAM(p * 4, 8 * p, 4 * p, m / 2, P_compute_BO);

      SB_STRIDE(8, 8);

      complex<int16_t> *_c = c + io * p;
      complex<int16_t> *_a = a + io * m;

      SB_CONTEXT_OFFSET(15, p);
      SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
      SB_CONTEXT_OFFSET(15, m)
      SB_REPEAT_PORT(p / 4);
      SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
      SB_CONTEXT_OFFSET(15, 0);

    /*for (int i = _n; i < n; ++i) {
      SB_CONTEXT_I(1 << (i - _n));
      SB_DMA_WRITE(P_compute_O, 0, 4 * p, 1, c + i * p);
      SB_REPEAT_PORT(p / 4);
      SB_DMA_READ(a + i * m, 0, 4 * m, 1, P_compute_A);
    }*/

    //SB_STRIDE(8, 8);

    //complex<int16_t> *_c = c + _n * p;
    //complex<int16_t> *_a = a + _n * m;

    //SB_CONTEXT_I(1);
    //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
    //SB_REPEAT_PORT(p / 4);
    //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
    //_c += p;
    //_a += m;

    //SB_CONTEXT_I(2);
    //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
    //SB_REPEAT_PORT(p / 4);
    //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
    //_c += p;
    //_a += m;

    //SB_CONTEXT_I(4);
    //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
    //SB_REPEAT_PORT(p / 4);
    //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
    //_c += p;
    //_a += m;

    //SB_CONTEXT_I(8);
    //SB_DMA_WRITE_SIMP(P_compute_O, p / 2, _c);
    //SB_REPEAT_PORT(p / 4);
    //SB_DMA_READ_SIMP(_a, m / 2, P_compute_A);
  }

  SB_CONTEXT(255);
  SB_WAIT_ALL();
}
