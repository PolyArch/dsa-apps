#include "Fx16ComplexMul.dfg.h"
#include <iostream>
#include "sb_insts.h"
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
  complex<int16_t> a(DOUBLE_TO_FIX(.1), DOUBLE_TO_FIX(2));
  complex<int16_t> b(DOUBLE_TO_FIX(-.03), DOUBLE_TO_FIX(4));

  complex<int16_t> array1[] = {a, b};
  complex<int16_t> array2[] = {b, a};

  complex<int16_t> res_a(complex_mul(a, b));
  complex<int16_t> res_b(complex_mul(b, a));
  complex<int16_t> ans[] = {res_a, res_b};
  //complex<int16_t> real[2];
  //complex<int16_t> imag[2];
  complex<int16_t> res[2];

  SB_CONFIG(Fx16ComplexMul_config, Fx16ComplexMul_size);
  SB_DMA_READ(array1, 8, 8, 1, P_Fx16ComplexMul_A);
  SB_DMA_READ(array2, 8, 8, 1, P_Fx16ComplexMul_B);
  SB_DMA_WRITE(P_Fx16ComplexMul_O, 8, 8, 1, res);
  SB_WAIT_ALL();

  //for (int i = 0; i < 2; ++i)
    //std::cout << FIX_TO_DOUBLE(res[i].real()) << ", " << FIX_TO_DOUBLE(res[i].imag()) << "\n";
  //for (int i = 0; i < 2; ++i)
    //std::cout << FIX_TO_DOUBLE(ans[i].real()) << ", " << FIX_TO_DOUBLE(ans[i].imag()) << "\n";

  compare("Fx16ComplexMul", res, ans, 2);

  return 0;
}
