#include "qr.h"
#include "fileop.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"

using std::complex;

complex<float> a[N * N], Q[N * N], R[N * N];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N * N, a);

  begin_roi();
  qr(a, Q, R);
  end_roi();
  sb_stats();

  if (!compare_n_float_complex(ref_data, N * N, Q) || !compare_n_float_complex(ref_data, N * N, R))
    return 1;

  puts("result correct!");
  return 0;
}
