#include <complex>
#include <cmath>
#include <algorithm>
#include "softbrain-config/fixed_point.h"
#include "sb_insts.h"
#include "compute.h"

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

void transform(int n, int m, complex<int16_t> *a, complex<int16_t> *b, complex<int16_t> *c) {
  SB_CONFIG(compute_config, compute_size);
  SB_DMA_READ(a, 8 * m, 4 * m, n / 2, P_compute_AE);
  SB_DMA_READ(a + m, 8 * m, 4 * m, n / 2, P_compute_AO);
  SB_DMA_READ(b, 0, 4 * m, n / 2, P_compute_B);
  SB_2D_CONST(P_compute_reset, 2, m / 2 - 1, 1, 1, n / 2);
  SB_DMA_WRITE(P_compute_O, 8, 8, n / 2, c);
  SB_WAIT_ALL();
}
