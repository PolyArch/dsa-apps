#include "cholesky.h"
#include "fileop.h"
#include <complex.h>
#include <iostream>

complex<float> a[N * N], L[N * N];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N * N, a);

  begin_roi();
  cholesky(a, L);
  end_roi();
  sb_stats();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << L[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  if (!compare_n_float_complex(ref_data, N * N, L))
    return 1;

  puts("result correct!");
  return 0;
}
