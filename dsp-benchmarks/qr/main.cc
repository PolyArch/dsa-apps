#include "qr.h"
#include "fileop.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include <iostream>

using std::complex;

complex<float> a[_N_ * _N_], Q[_N_ * _N_], R[_N_ * _N_];
complex<float> aa[_N_ * _N_];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  std::cout << std::fixed;
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, _N_ * _N_, a);

  qr(aa, aa, aa);
  begin_roi();
  qr(a, Q, R);
  end_roi();
  sb_stats();

  /*for (int i = 0; i < _N_; ++i) {
    for (int j = 0; j < _N_; ++j) {
      complex<float> sum(0);
      for (int k = 0; k < _N_; ++k)
        sum += Q[i * _N_ + k] * std::conj(Q[j * _N_ + k]);
      if (i == j && (fabs(sum.real() - 1.0) > eps || fabs(sum.imag()) > eps)) {
        puts("Q is not a orthogonal matrix!");
        printf("%f %f\n", sum.real(), sum.imag());
        return 1;
      }
      if (i != j && fabs(sum.real() + sum.imag()) > 2 * eps) {
        puts("Q is not a orthogonal matrix!");
        printf("%f %f\n", sum.real(), sum.imag());
        return 1;
      }
    }
  }

  for (int i = 0; i < _N_ * _N_; ++i) aa[i] = 0;

  for (int i = 0; i < _N_; ++i)
    for (int j = 0; j < _N_; ++j)
      for (int k = 0; k < _N_; ++k)
        aa[i * _N_ + j] += Q[i * _N_ + k] * R[k * _N_ + j];

  if (!compare_n_float_complex(ref_data, _N_ * _N_, aa)) {
    puts("error origin matrix");
    return 1;
  }*/
  /*if (!compare_n_float_complex(ref_data, _N_ * _N_, R)) {
    puts("error r");
    return 1;
  }*/

  //puts("result correct!");
  return 0;
}
