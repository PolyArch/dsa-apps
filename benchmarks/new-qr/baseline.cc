#include "qr.h"
#include <iostream>
#include "sim_timing.h"

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(complex<float> *a, complex<float> *q, complex<float> *r) {
  int i, j, k, x, y;
  complex<float> *tmp = new complex<float>[N * N];
  const complex<float> one(1, 0), zero(0, 0);
  for (i = 0; i < N; ++i) {
    q[i * (N + 1)] = 1;
  }
  for (i = 0; i < N * N; ++i) {
    r[i] = a[i];
  }
  for (i = 0; i < N; ++i) {

    complex<float> v[N - i];
    {
      complex<float> norm(0, 0);
      for (j = i; j < N; ++j) {
        v[j - i] = r[j * N + i];
        norm += v[j - i] * std::conj(v[j - i]);
      }
      norm = std::sqrt(norm);
      v[0] += std::exp(complex<float>(0, std::arg(v[0]))) * norm;
    }

    {
      complex<float> norm(0, 0);
      for (j = i; j < N; ++j)
        norm += v[j - i] * std::conj(v[j - i]);
      norm = std::sqrt(norm);
      for (j = i; j < N; ++j)
        v[j - i] /= norm;
    }

    complex<float> w;
    {
      complex<float> xv(0, 0), vx(0, 0);
      for (j = i; j < N; ++j) {
        xv += std::conj(r[j * N + i]) * v[j - i];
        vx += std::conj(v[j - i]) * r[j * N + i];
      }
      w = one + xv / vx;
    }

    {
      for (y = 0; y < N; ++y) {
        for (x = i; x < N; ++x) {
          tmp[y * N + x] = 0;
          for (k = i; k < N; ++k) {
            tmp[y * N + x] += q[y * N + k] * ((k == x ? one : zero) - v[k - i] * std::conj(v[x - i]) * w);
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
            tmp[y * N + x] += ((y == k ? one : zero) - v[y - i] * std::conj(v[k - i]) * w) * r[k * N + x];
        }
      }
    }
    for (y = i; y < N; ++y) for (x = i; x < N; ++x) {
      r[y * N + x] = tmp[y * N + x];
    }

  }
}
#undef h

