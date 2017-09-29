#include "filter.h"
#include <complex.h>
#include <iostream>

complex<float> a[N], b[FILTER], c[N - FILTER + 1], w[(N + FILTER) << 1];

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
  for (int i = 0; i < FILTER; ++i) {
    float real, imag;
    fscanf(input_data, " (%f+%fj)", &real, &imag);
    b[i] = complex<float>(real, imag);
  }


  begin_roi();
  filter(a, b, c);
  end_roi();

  for (int i = 0; i < N - FILTER + 1; ++i) {
    float real, imag;
    fscanf(ref_data, " (%f+%fj)", &real, &imag);
    if (fabs(real - c[i].real()) + fabs(imag - c[i].imag()) > eps * 2) {
      printf("error @%d %d\n", i / N, i % N);
      printf("%f+%fi %f+%fi\n", real, imag, c[i].real(), c[i].imag());
      return 1;
    }
  }
  puts("result correct!");
  return 0;
}
