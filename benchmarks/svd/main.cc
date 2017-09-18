#include "svd.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"

using std::complex;

complex<float> a[N * N], U[N * N], S[N], V[N * N];

bool compare(complex<float> *a, int n, FILE *ref_data) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(ref_data, " (%f+%fj)", &real, &imag);
    if (fabs(real - a[i].real()) + fabs(imag - a[i].imag()) > eps * 20) {
      printf("error @%d %d\n", i / N, i % N);
      printf("%f+%fi %f+%fi\n", real, imag, a[i].real(), a[i].imag());
      return false;
    }
  }
  return true;
}

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  for (int i = 0; i < N * N; ++i) {
    float real, imag;
    fscanf(input_data, " (%f+%fj)", &real, &imag);
    a[i] = complex<float>(real, imag);
  }

  begin_roi();
  svd(a, U, S, V);
  end_roi();

  if (U[0].real() < 0) {
    for (int i = 0; i < N * N; ++i) {
      U[i] = -U[i];
      V[i] = -V[i];
    }
  }

  if (!compare(U, N * N, ref_data)) {
    puts("Error U!");
    return 1;
  }
  if (!compare(S, N, ref_data)) {
    puts("Error \\sigma!");
    return 1;
  }
  if (!compare(V, N * N, ref_data)) {
    puts("Error V*!");
    return 1;
  }

  puts("result correct!");
  return 0;
}
