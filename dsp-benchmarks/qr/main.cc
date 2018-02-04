#include "qr.h"
#include "fileop.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include <iostream>

using std::complex;

complex<float> a[N * N], Q[N * N], R[N * N];
complex<float> aa[N * N];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  std::cout << std::fixed;
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N * N, a);

  qr(aa, aa, aa);
  begin_roi();
  qr(a, Q, R);
  end_roi();
  sb_stats();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex<float> sum(0);
      for (int k = 0; k < N; ++k)
        sum += Q[i * N + k] * std::conj(Q[j * N + k]);
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

  for (int i = 0; i < N * N; ++i) aa[i] = 0;

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        aa[i * N + j] += Q[i * N + k] * R[k * N + j];

  if (!compare_n_float_complex(ref_data, N * N, aa)) {
    puts("error origin matrix");
    return 1;
  }
  /*if (!compare_n_float_complex(ref_data, N * N, R)) {
    puts("error r");
    return 1;
  }*/

  puts("result correct!");
  return 0;
}
