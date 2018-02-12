#include "fft.h"
#include "fileop.h"
#include <string.h>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#define PI 3.14159265358979303
#include <iostream>

using std::complex;

complex<float> a[_N_], _a[_N_], w[_N_ / 2];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, _N_, a);

  for (int i = 0; i < _N_ / 2; ++i) {
    w[i] = complex<float>(cos(2 * PI * i / _N_), sin(2 * PI * i / _N_));
  }

  fft(_a, w);
  begin_roi();
  fft(a, w);
  end_roi();
  sb_stats();

  int N = _N_;

  //for (int i = 0; i < _N_; ++i)
    //std::cout << a[i] << (i == (N - 1) ? "\n" : " ");

  puts("I hope the result is correct because the order of the result is wierd now...");

  return 0;
}
