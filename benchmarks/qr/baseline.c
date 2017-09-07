#include "qr.h"
#include "sim_timing.h"

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(float *a, float *q, float *r) {
  int i, j, k, x, y; float *tmp = (float *) malloc(N * N * sizeof(float));
  float *v = (float *) malloc(N * sizeof(float));
  for (i = 0; i < N; ++i) {
    q[i * (N + 1)] = 1;
  }
  for (i = 0; i < N * N; ++i) {
    r[i] = a[i];
  }
  for (i = 0; i < N; ++i) {
    float dot = 0.;

    {
      for (j = i; j < N; ++j) {
        v[j] = r[j * N + i];
        dot += v[j] * v[j];
      }
    }

    v[i] += (r[i * (N + 1)] < -eps ? -1 : 1) * sqrt(dot);

    dot = 0.;
    for (j = i; j < N; ++j)
      dot += v[j] * v[j];

    dot = sqrt(dot);
    for (j = i; j < N; ++j)
      v[j] /= dot;

    {
      for (y = 0; y < N; ++y) {
        for (x = i; x < N; ++x) {
          tmp[y * N + x] = 0;
          for (k = i; k < N; ++k) {
            tmp[y * N + x] += q[y * N + k] * h(k, x);
          }
        }
      }
    }

    for (y = 0; y < N; ++y) for (x = i; x < N; ++x) {
      q[y * N + x] = tmp[y * N + x];
    }

    {
      for (y = i; y < N; ++y) {
        for (x = i; x < N; ++x) {
          tmp[y * N + x] = 0;
          for (k = i; k < N; ++k)
            tmp[y * N + x] += h(y, k) * r[k * N + x];
        }
      }
    }
    for (y = i; y < N; ++y) for (x = i; x < N; ++x) {
      r[y * N + x] = tmp[y * N + x];
    }

  }
}
#undef h

