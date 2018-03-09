#include "mkl.h"
#include "fileop.h"
#include <cstring>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include <iostream>

using std::complex;

complex<float> a[_N_ * _N_], U[_N_ * _N_], V[_N_ * _N_], tmp[_N_ * _N_], res[_N_ * _N_];
complex<float> aa[_N_ * _N_];
complex<float> _one(1), _zero(0);
float s[_N_], ss[_N_], superb[_N_];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  std::cout << std::fixed;
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  int n = _N_;

  read_n_float_complex(input_data, n * n, a);
  memcpy(aa, a, sizeof a);

  LAPACKE_cgesvd(
      CblasRowMajor, 'A', 'A',
      n, n,
      (lapack_complex_float *) aa, n,
      ss,
      (lapack_complex_float *) aa, n,
      (lapack_complex_float *) aa, n,
      superb
  );
  begin_roi();
  LAPACKE_cgesvd(
      CblasRowMajor, 'A', 'A',
      n, n,
      (lapack_complex_float *) a, n,
      s,
      (lapack_complex_float *) U, n,
      (lapack_complex_float *) V, n,
      superb
  );
  end_roi();
  sb_stats();

  puts("I hope the result is correct, because MKL has different results...");
  return 0;
}
