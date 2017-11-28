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

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  std::cout << std::fixed;
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N * N, a);

  qr(a, Q, R);
  begin_roi();
  qr(a, Q, R);
  end_roi();
  sb_stats();

  //for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) std::cout << R[i * N + j] << "\n";

  if (!compare_n_float_complex(ref_data, N * N, Q)) {
    puts("error q");
    return 1;
  }
  if (!compare_n_float_complex(ref_data, N * N, R)) {
    puts("error r");
    return 1;
  }

  puts("result correct!");
  return 0;
}
