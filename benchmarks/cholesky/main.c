#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>

#define N 64
#define eps 1e-4

float a[N * N] = {6, 11, 11, 26}, L[N * N];

void cholesky(float *a, float *L) {
  float *A = (float *) malloc(N * N * sizeof(float));
  for (int i = 0; i < N * N; ++i) {
    A[i] = a[i];
  }
  for (int i = 0; i < N; ++i) {
    float div = sqrt(A[i * (N + 1)]);
    float *b = A + i * (N + 1);
    L[i * (N + 1)] = div;
    for (int j = i + 1; j < N; ++j) {
      L[j * N + i] = b[j - i] / div;
    }
    for (int j = i + 1; j < N; ++j) {
      for (int k = i + 1; k < N; ++k) {
        A[j * N + k] -= b[j - i] * b[k - i] / A[i * (N + 1)];
      }
    }
    for (int j = i + 1; j < N; ++j)
      A[i * N + j] = A[j * N + i] = 0;
    A[i * (N + 1)] = 1;
    /*puts("A:");
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%f ", A[j * N + k]);
      }
      puts("");
    }
    printf("L: ");
    for (int j = i; j < N; ++j)
      printf("%f ", L[j * N + i]);
    puts("");*/
  }
}

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      fscanf(input_data, "%f", a + i * N + j);
    }
  }
  cholesky(a, L);
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
