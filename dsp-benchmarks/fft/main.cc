#include "fft.h"
#include "fileop.h"
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#define PI 3.14159265358979303

using std::complex;

complex<float> a[N], w[N / 2];

bool compare(complex<float> *a, int n, FILE *ref_data) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(ref_data, "%f%f", &real, &imag);
    if (fabs(real - a[i].real()) + fabs(imag - a[i].imag()) > eps) {
      printf("expect %f+%fi but %f+%fi\n", real, imag, a[i].real(), a[i].imag());
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

  read_n_float_complex(input_data, N, a);

  for (int i = 0; i < N / 2; ++i) {
    w[i] = complex<float>(cos(2 * PI * i / N), sin(2 * PI * i / N));
  }

  begin_roi();
  fft(a, w);
  end_roi();
  sb_stats();

  if (!compare_n_float_complex(ref_data, N, a)) {
    puts("Error result!");
    return 1;
  }

  puts("result correct!");
  return 0;
}
