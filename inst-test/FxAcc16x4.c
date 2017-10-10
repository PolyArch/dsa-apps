#include "FxAcc16x4.h"
#include "sb_insts.h"
#include "softbrain-config/fixed_point.h"
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
  complex<int16_t> a(DOUBLE_TO_FIX(1), DOUBLE_TO_FIX(2));
  complex<int16_t> b(DOUBLE_TO_FIX(3), DOUBLE_TO_FIX(4));

  complex<int16_t> array1[] = {a, b};

  complex<int16_t> acc_a(complex_add(a, a));
  acc_a = complex<int16_t> (complex_add(acc_a, a));

  complex<int16_t> acc_b(complex_add(b, b));
  acc_b = complex<int16_t> (complex_add(acc_b, b));

  complex<int16_t> ans[] = {acc_a, acc_b};
  complex<int16_t> res[2];

  SB_CONFIG(FxAcc16x4_config, FxAcc16x4_size);
  SB_DMA_READ(array1, 0, 8, 3, P_FxAcc16x4_A);
  SB_CONST(P_FxAcc16x4_reset, 0, 3);
  SB_DMA_WRITE(P_FxAcc16x4_B, 0, 8, 3, res);
  SB_WAIT_ALL();
  compare("FxAcc16x4", res, ans, 2);


  return 0;
}
