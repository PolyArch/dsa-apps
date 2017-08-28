#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

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
      float *bp = b + 1;
      for (int j = i + 1; j < N; ++j) {
        L[j * N + i] = *bp++ / div;
      }
    }
    {
      float *bj = b + 1, *bk;
      float v = A[i * (N + 1)];
      char delta = 0;
      for (int j = i + 1; j < N; ++j) {
        bk = bj;
        for (int k = j; k < N; ++k) {
          A[j * N + k] -= *bj * (*bk++) / v;
        }
        ++bj;
      }
    }
  }
}

