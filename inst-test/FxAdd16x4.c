#include "FxAdd16x4.dfg.h"
#include "ss_insts.h"
#include "ss-config/fixed_point.h"
#include "check.h"
#include <complex>
#include <stdint.h>

using std::complex;

#define complex_mul(a, b) \
  FIX_MINUS(FIX_MUL((a).real(), (b).real()), FIX_MUL((a).imag(), (b).imag())), \
  FIX_ADD(FIX_MUL((a).real(), (b).imag()), FIX_MUL((a).imag(), (b).real()))

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) FIX_ADD((a).real(), (b).real()), FIX_ADD((a).imag(), (b).imag())

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

int main() {
  complex<int16_t> a(DOUBLE_TO_FIX(-0.01), DOUBLE_TO_FIX(2));
  complex<int16_t> b(DOUBLE_TO_FIX(-0.023), DOUBLE_TO_FIX(4));

  complex<int16_t> array1[] = {a, b};
  complex<int16_t> array2[] = {b, a};

  complex<int16_t> res_a(complex_add(a, b));
  complex<int16_t> ans[] = {res_a, res_a};
  complex<int16_t> res[2];

  SS_CONFIG(FxAdd16x4_config, FxAdd16x4_size);
  SS_DMA_READ(array1, 8, 8, 1, P_FxAdd16x4_A);
  SS_DMA_READ(array2, 8, 8, 1, P_FxAdd16x4_B);
  SS_DMA_WRITE(P_FxAdd16x4_C, 8, 8, 1, res);
  SS_WAIT_ALL();
  compare("FxAdd16x4", res, ans, 2);

  return 0;
}
