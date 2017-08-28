#include "cholesky.h"

float a[N * N], L[N * N];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      fscanf(input_data, "%f", a + i * N + j);
    }
  }
  begin_roi();
  cholesky(a, L);
  end_roi();
  for (int i = 0; i < N * N; ++i) {
    float value;
    fscanf(ref_data, "%f", &value);
    if (fabs(value - L[i]) > eps) {
      printf("error @%d %d\n", i / N, i % N);
      printf("%f %f\n", value, L[i]);
      return 1;
    }
  }
  puts("result correct!");
  return 0;
}
