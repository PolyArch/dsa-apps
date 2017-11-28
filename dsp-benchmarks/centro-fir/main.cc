#include "filter.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include "fileop.h"
#include "softbrain-config/fixed_point.h"
#define PI 3.14159265358979303

using std::complex;

complex<float> a[N], b[M], c[N - M + 1];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N, a);
  read_n_float_complex(input_data, M, b);

  begin_roi();
  filter(N, M, a, b, c);
  end_roi();
  sb_stats();

  if (!compare_n_float_complex(ref_data, N - M + 1, c))
    return 1;

  puts("result correct!");
  return 0;
}
