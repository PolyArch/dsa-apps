#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "compute.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"

#define N 64
#define eps 1e-4

float a[N * N], L[N * N];

void cholesky(float *a, float *L) {
  const int n = N + 2 | 1;
  float *A = (float *) malloc(n * n * sizeof(float));

  SB_CONFIG(compute_config, compute_size);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * n + j] = a[i * N + j];
    }
  }
  for (int i = 0; i < N; ++i) {
    float div = sqrt(A[i * (n + 1)]);
    //float *b = A + i * (n + 1);
    float b[N - i + 1];
    float bb[N - i];
    L[i * (N + 1)] = div;
    {
      float *_b = A + i * (n + 1);
      float *bp = b;
      *bp++ = *_b++;
      float *bbp = bb;
      for (int j = i + 1; j < N; ++j) {
        float v = *_b++;
        *bp++ = v;
        *bbp++= v;
        L[j * N + i] = v / div;
      }
      *bp = *bbp = 0;
    }
    //begin_roi();
    {
      float *bj = b + 1, *bk;
      //float v = A[i * (n + 1)];
      float v = 1. / A[i * (n + 1)];
      char delta = 0;
      for (int j = i + 1; j < N; ++j) {

        union {
          float a[2];
          uint64_t v;
        } ri_bj = {*bj, *bj}, ri_v = {-v, -v};

        int times = (N - j - 1 >> 1) + 1;
        //times = 1;
        //printf("%d,%d: %d\n", i, j, times);
        delta ^= 1;
        SB_CONST(P_compute_A, ri_bj.v, times);
        SB_DMA_READ(delta ? bb + j - i - 1 : b + j - i, 8, 8, times, P_compute_B);
        SB_CONST(P_compute_C, ri_v.v, times);
        SB_DMA_READ(A + (n + 1) * j, 8, 8, times, P_compute_D);
        SB_DMA_WRITE(P_compute_G, 8, 8, times, A + (n + 1) * j);

        /*
        bk = b + j - i;
        for (int k = j; k < N; ++k) {
          A[j * n + k] -= *bj * (*bk++) / v;
        }*/
        ++bj;
      }
      SB_WAIT_ALL();
    }
    //end_roi();
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
