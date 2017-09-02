#include "cholesky.h"
#include <complex.h>
#include <iostream>

complex<float> a[N * N], L[N * N];

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
    //printf("%f %f\n", real, imag);
  }
  begin_roi();
  cholesky(a, L);
  end_roi();
  /*for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << L[i * N + j] << " ";
    }
    std::cout << "\n";
  }*/
  for (int i = 0; i < N * N; ++i) {
    float real, imag;
    fscanf(ref_data, " (%f+%fj)", &real, &imag);
    if (fabs(real - L[i].real()) + fabs(imag - L[i].imag()) > eps * 2) {
      printf("error @%d %d\n", i / N, i % N);
      printf("%f+%fi %f+%fi\n", real, imag, L[i].real(), L[i].imag());
      return 1;
    }
  }
  puts("result correct!");
  return 0;
}
