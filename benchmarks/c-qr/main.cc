#include "qr.h"
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
  for (int i = 0; i < N * N; ++i) {
    float real, imag;
    fscanf(input_data, " (%f+%fj)", &real, &imag);
    a[i] = complex<float>(real, imag);
  }
  begin_roi();
  qr(a, Q, R);
  end_roi();
  for (int i = 0; i < N * N; ++i) {
    float real, imag;
    fscanf(ref_data, " (%f+%fj)", &real, &imag);
    if (fabs(real - Q[i].real()) + fabs(imag - Q[i].imag()) > eps * 2) {
      printf("error @%d %d\n", i / N, i % N);
      printf("%f+%fi %f+%fi\n", real, imag, Q[i].real(), Q[i].imag());
      return 1;
    }
  }
  for (int i = 0; i < N * N; ++i) {
    float real, imag;
    fscanf(ref_data, " (%f+%fj)", &real, &imag);
    if (fabs(real - R[i].real()) + fabs(imag - R[i].imag()) > eps * 2) {
      printf("error @%d %d\n", i / N, i % N);
      printf("%f+%fi %f+%fi\n", real, imag, R[i].real(), R[i].imag());
      return 1;
    }
  }
  puts("result correct!");
  return 0;
}
