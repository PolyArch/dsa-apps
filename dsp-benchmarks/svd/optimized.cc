#include "svd.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

complex<float> f[N], d[N], r[N * N], temp[N];

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

void svd(complex<float> *a, complex<float> *u, complex<float> *s, complex<float> *v) {
  for (int i = 0; i < N - 1; ++i) {
    int len = N - i;
    complex<float> hv[len], alpha;
    for (int j = 0; j < len; ++j)
      hv[j] = (i ? r : a)[j * len];
    household(hv, len, d[i]);

    for (int j = 1; j < len; ++j) {
      temp[j] = 0;
      for (int k = 0; k < len; ++k) {
        temp[j] += complex<float>(complex_conj_mul(hv[k], (i ? r : a)[k * len + j]));
      }
      temp[j] = complex<float>(temp[j].real() * 2, temp[j].imag() * 2);
      //std::cout << temp[j] << "\n";
    }

    for (int j = 0; j < len; ++j) {
      for (int k = 1; k < len; ++k) {
        complex<float> delta(complex_mul(temp[k], hv[j]));
        r[j * (len - 1) + (k - 1)] = complex<float>(complex_sub((i ? r : a)[j * len + k], delta));
      }
    }

    if (i != N - 2) {
      --len;
      for (int j = 0; j < len; ++j)
        hv[j] = r[j];
      household(hv, len, f[i]);

      for (int j = 0; j < len; ++j) {
        temp[j] = 0;
        for (int k = 0; k < len; ++k) {
          temp[j] += complex<float>(complex_conj_mul(hv[k], r[(j + 1) * len + k]));
        }
        temp[j] = complex<float>(temp[j].real() * 2, temp[j].imag() * 2);
      }
      for (int j = 0; j < len; ++j) {
        for (int k = 0; k < len; ++k) {
          complex<float> delta(complex_mul(temp[j], hv[k]));
          r[j * len + k] = complex<float>(complex_sub(r[(j + 1) * len + k], delta));
        }
      }
      if (!i) {
        v[0] = complex<float>(1, 0);
        for (int j = 1; j < N; ++j) {
          v[j] = v[j * N] = complex<float>(0, 0);
          for (int k = 1; k < N; ++k) {
            complex<float> delta(complex_conj_mul(hv[j - 1], hv[k - 1]));
            complex<float> diag(j == k, 0);
            complex<float> val(delta.real() * 2, delta.imag() * 2);
            v[j * N + k] = complex<float>(complex_sub(diag, val));
          }
        }
        //for (int j = 0; j < N; ++j) { for (int k = 0; k < N; ++k) std::cout << v[j * N + k] << " "; std::cout << "\n"; }
      } else {
        for (int k = 1; k < N; ++k) {
          temp[k] = 0;
          for (int j = i + 1; j < N; ++j) {
            complex<float> delta(complex_mul(hv[j - i - 1], v[k * N + j]));
            temp[k] = complex<float>(complex_add(temp[k], delta));
          }
          temp[k] = complex<float>(temp[k].real() * 2, temp[k].imag() * 2);
        }
        //for (int j = 1; j < N; ++j) std::cout << temp[j] << " "; std::cout << "\n";
      }
    }

  }
  f[N - 2] = r[0];
  d[N - 1] = r[1];

  //for (int i = 1; i < N; ++i) std::cout << f[i - 1] << " "; std::cout << "\n";
  //for (int i = 0; i < N; ++i) std::cout << d[i] << " "; std::cout << "\n";
}
