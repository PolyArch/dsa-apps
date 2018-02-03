#include "svd.h"
#include "fileop.h"
#include <algorithm>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"

using std::complex;

complex<float> a[N * N], U[N * N], V[N * N], tmp[N * N], res[N * N];
complex<float> ss[N];
float S[N];
complex<float> aa[N * N];

int main() {
  FILE *input_data = fopen("input.data", "r");
  FILE *ref_data = fopen("ref.data", "r");

  if (!input_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N * N, a);

  //svd(aa, aa, aa, aa);
  begin_roi();
  svd(a, U, S, V);
  end_roi();
  sb_stats();

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      tmp[i * N + j] = U[i * N + j] * S[j];

  for (int i = 0; i < N; ++i)
     for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        res[i * N + j] += tmp[i * N + k] * V[k * N + j];

  std::sort(S, S + N, [](float a, float b) { return a > b; });
  for (int i = 0; i < N; ++i)
    ss[i] = S[i];

  if (!compare_n_float_complex(ref_data, N, ss)) {
    puts("singular value error!");
    return 0;
  }

  if (!compare_n_float_complex(ref_data, N * N, res)) {
    puts("origin matrix error!");
    return 0;
  }

  puts("result correct!");

  return 0;
}
