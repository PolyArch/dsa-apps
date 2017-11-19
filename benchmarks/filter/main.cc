#include "filter.h"
#include "fileop.h"
#include <complex.h>
#include <iostream>

complex<float> a[N], b[FILTER], c[N - FILTER + 1], w[(N + FILTER) << 1];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N, a);
  read_n_float_complex(input_data, N, b);

  begin_roi();
  filter(a, b, c);
  end_roi();
  sb_stats();

  if (!compare_n_float_complex(ref_data, N - FILTER + 1, c))
    return 1;

  puts("result correct!");
  return 0;
}
