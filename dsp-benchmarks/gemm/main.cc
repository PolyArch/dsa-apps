#include "gemm.h"
#include "fileop.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include "ss-config/fixed_point.h"
#define PI 3.14159265358979303

using std::complex;

complex<int16_t> a[N * M], b[M * P], c[N * P], cc[N * P];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_fix_complex(input_data, N * M, a);
  read_n_fix_complex(input_data, P * M, b);

  gemm(N, M, P, a, b, cc);
  begin_roi();
  gemm(N, M, P, a, b, c);
  end_roi();
  sb_stats();

  if (!compare_n_fix_complex( ref_data, N * P, c)) {
    puts("Error result!");
    return 1;
  }

  puts("result correct!");
  return 0;
}
