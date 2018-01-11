#include "fft.h"
#include "fileop.h"
#include <string.h>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#define PI 3.14159265358979303

using std::complex;

complex<float> a[N], _a[N], w[N / 2];
complex<float> b[N], _b[N];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N, a);

  for (int i = 0; i < N / 2; ++i) {
    w[i] = complex<float>(cos(2 * PI * i / N), sin(2 * PI * i / N));
  }

  fft(_a, _b, w);
  begin_roi();
  complex<float> *res = fft(a, b, w);
  end_roi();
  sb_stats();

  if (!compare_n_float_complex(ref_data, N, res)) {
    puts("Error result!");
    return 1;
  }

  puts("result correct!");
  return 0;
}
