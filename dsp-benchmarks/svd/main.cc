#include "svd.h"
#include "fileop.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"

using std::complex;

complex<float> a[N * N], U[N * N], S[N], V[N * N], tmp[N * N], res[N * N];
complex<float> aa[N * N];

int main() {
  FILE *input_data = fopen("input.data", "r");
  FILE *ref_data = fopen("input.data", "r");

  if (!input_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, N * N, a);

  svd(aa, aa, aa, aa);
  begin_roi();
  svd(a, U, S, V);
  end_roi();
  sb_stats();

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      tmp[i * N + j] += U[i * N + j] * S[j];

  for (int i = 0; i < N; ++i)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        res[i * N + j] += tmp[i * N + k] * std::conj(V[j * N + k]);

  if (compare_n_float_complex(ref_data, N * N, res))
    puts("result correct!");

  return 0;
}
