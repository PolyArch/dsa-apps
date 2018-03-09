#include "qr.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "matvec.h"

complex<float> buffer[1024];

void qr(complex<float> *a, complex<float> *tau) {
  int N = _N_;
  complex<float> *w = buffer;
  complex<float> *v = buffer + N;

  for (int i = 0; i < N - 1; ++i) {
    int n = N - i;
    float normx = 0;
    for (int j = i; j < N; ++j) {
      w[j - i] = a[j * N + i];
      normx += complex_norm(w[j - i]);
    }
    normx = sqrt(normx);
    float norm0 = 1. / sqrt(complex_norm(w[0]));
    complex<float> s = -w[0] * norm0;
    a[i * N + i] = s * normx;
    complex<float> u1 = 1.0f / (w[0] - s * normx);
    w[0] = 1.0f;
    for (int j = i + 1; j < N; ++j) {
      w[j - i] *= u1;
      a[j * N + i] = w[j - i];
    }
    tau[i] = -std::conj(s) / u1 / normx;
    //householder done

    CPUvec_mul_mat(a + i * N + i + 1, n, n - 1, N, w, true, v);
    for (int j = i; j < N; ++j) {
      for (int k = i + 1; k < N; ++k) {
        a[j * N + k] -= tau[i] * w[j - i] * v[k - i - 1];
      }
    }
  }
}

void unitary(complex<float> *a, complex<float> *tau, complex<float> *q) {
  int N = _N_;
  q[N * N - 1] = 1.0f;
  complex<float> *w = buffer;
  complex<float> *v = buffer + N;
  for (int i = N - 2; i >= 0; --i) {
    int n = N - i;
    for (int j = i + 1; j < N; ++j)
      w[j - i - 1] = a[j * N + i];
    CPUvec_mul_mat(q + (i + 1) * N + i + 1, n - 1, n - 1, N, w, true, v);
    //for (int j = 0; j < n - 1; ++j) std::cout << v[j] << " "; std::cout << "\n";
    q[i * N + i] = 1.0f - tau[i];
    for (int j = i + 1; j < N; ++j) {
      q[i * N + j] = -tau[i] * v[j - i - 1];
      q[j * N + i] = -tau[i] * a[j * N + i];
      for (int k = i + 1; k < N; ++k) {
        q[j * N + k] -= tau[i] * w[j - i - 1] * v[k - i - 1];
      }
    }
    //for (int j = i; j < N; ++j) {
    //  for (int k = i; k < N; ++k)
    //    std::cout << q[j * N + k] << " ";
    //  std::cout << "\n";
    //}
  }
}
