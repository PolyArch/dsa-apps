#include "cholesky.h"
#include "fileop.h"
#include <complex.h>
#include <iostream>
#include "sim_timing.h"

complex<float> a[N * N], L[N * N], aa[N * N];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N * N, a);

  cholesky(aa, aa);
  begin_roi();
  cholesky(a, L);
  end_roi();
  sb_stats();

  if (!compare_n_float_complex(ref_data, N * N, L))
      return 1;

  puts("result correct!");
  return 0;
}
