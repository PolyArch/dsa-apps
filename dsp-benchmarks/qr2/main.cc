#include "qr.h"
#include "fileop.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include <iostream>

using std::complex;

complex<float> a[_N_ * _N_], tau[_N_], q[_N_ * _N_];
complex<float> aa[_N_ * _N_];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  std::cout << std::fixed;
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, _N_ * _N_, a);

  //qr(aa, aa);
  //unitary(aa, aa, aa);
  begin_roi();
  qr(a, tau);
  //unitary(a, tau, q);
  end_roi();
  sb_stats();
  
  if (!compare_n_float_complex(ref_data, _N_ * _N_, a)) {
    puts("error r");
    return 1;
  }

  if (!compare_n_float_complex(ref_data, _N_ - 1, tau)) {
    puts("error tau");
    return 1;
  }

  /*if (!compare_n_float_complex(ref_data, _N_ * _N_, q)) {
    puts("error q");
    return 1;
  }*/

  puts("result correct!");
  return 0;
}
