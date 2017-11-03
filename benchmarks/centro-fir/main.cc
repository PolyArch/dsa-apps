#include "filter.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include "softbrain-config/fixed_point.h"
#define PI 3.14159265358979303

using std::complex;

complex<float> a[N], b[M], c[N - M + 1];

bool compare(complex<float> *a, int n, FILE *ref_data) {
  for (int i = 0; i < n; ++i) {
    float real, imag, norm;
    fscanf(ref_data, " (%f+%fj)", &real, &imag);
    norm = real * real + imag * imag;
    if (((fabs(real - a[i].real())) + fabs(imag - a[i].imag())) / norm  > eps) {
      printf("@%d: expect %f+%fi but %f+%fi\n", i, real, imag, a[i].real(), a[i].imag());
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

  for (int i = 0; i < N; ++i) {
    float real, imag;
    fscanf(input_data, " (%f+%fj)", &real, &imag);
    a[i] = complex<float>(real, imag);
  }

  for (int i = 0; i < M; ++i) {
    float real, imag;
    fscanf(input_data, " (%f+%fj)", &real, &imag);
    b[i] = complex<float>(real, imag);
  }


  begin_roi();
  filter(N, M, a, b, c);
  end_roi();
  sb_stats();

  if (!compare(c, N - M + 1, ref_data)) {
    puts("Error result!");
    return 1;
  }

  puts("result correct!");
  return 0;
}
