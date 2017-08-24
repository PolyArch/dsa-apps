#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../../common/include/sim_timing.h"

#define N 64
#define eps 1e-4

float a[N * N], L[N * N];

void cholesky(float *a, float *L) {
  float *A = (float *) malloc(N * N * sizeof(float));
  for (int i = 0; i < N * N; ++i) {
    A[i] = a[i];
  }
  for (int i = 0; i < N; ++i) {
    float div = sqrt(A[i * (N + 1)]);
    float *b = A + i * (N + 1);
    L[i * (N + 1)] = div;
    {
      float *bj = b + 1;
      for (int j = i + 1; j < N; ++j) {
        L[j * N + i] = *bj++ / div;
      }
    }
    {
      float *bj = b + 1, *bk;
      float v = A[i * (N + 1)];
      for (int j = i + 1; j < N; ++j) {
        bk = b + j - i;
        for (int k = j; k < N; ++k) {
          A[j * N + k] -= *bj * (*bk++) / v;
        }
        ++bj;
      }
    }
  }
}

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
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
  /*for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) printf("%f ", L[i * N + j]);
    puts("");
  }*/
  puts("ok!");
  return 0;
}
