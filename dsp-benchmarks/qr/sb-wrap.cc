#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "matvec.h"

void household(complex<float> *v, int n, complex<float> &alpha) {
  float norm0 = complex_norm(v[0]), norm1 = 0;
  for (int j = 1; j < n; ++j) {
    norm1 += complex_norm(v[j]);
  }
  float _alpha = sqrt(1 + norm1 / norm0);
  alpha = complex<float>(-v->real() * _alpha, -v->imag() * _alpha);
  float rate = 1 + _alpha;
  v[0] = complex<float>(v->real() * rate, v->imag() * rate);
  norm1 += complex_norm(*v);
  norm1 = 1. / sqrt(norm1);
  for (int j = 0; j < n; ++j) {
    v[j] = complex<float>(v[j].real() * norm1, v[j].imag() * norm1);
  }
}

complex<float> temp[N];

void qr(complex<float> *a, complex<float> *q, complex<float> *r) {
  for (int i = 0; i < N; ++i) {
    int n  = N - i;
    complex<float> hv[n], alpha;
    for (int j = i; j < N; ++j) {
      hv[j - i] = (i ? r : a)[j * N + i];
    }
    household(hv, n, alpha);
    REVELvec_mul_mat(i, n, i, n, N, hv, true, (i ? r : a), temp);
    //for (int j = 0; j < n; ++j)
      //std::cout << temp[j] << (j == n - 1 ? "\n" : " ");
    for (int j = i; j < N; ++j) {
      for (int k = i; k < N; ++k) {
        r[j * N + k] = (i ? r : a)[j * N + k] - hv[j - i] * temp[k - i] * 2.0f;
      }
    }
    if (i) {
      REVELmat_mul_vec(0, N, i, n, N, q, hv, false, temp);
      for (int j = 0; j < N; ++j)
        for (int k = i; k < N; ++k)
          q[j * N + k] -= temp[j] * std::conj(hv[k - i]) * 2.0f;
    } else {
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
          q[j * N + k] = (j == k ? 1.0f : 0.0f) - std::conj(hv[k]) * hv[j] * 2.0f;
    }
  }
}
#undef h

