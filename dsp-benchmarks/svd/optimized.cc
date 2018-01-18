#include "svd.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_mul_cons(a, b) (a).real() * (b), (a).imag() * (b)

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

void household2(complex<float> &a, complex<float> &b, complex<float> &alpha) {
  float norm0 = complex_norm(a), norm1 = complex_norm(b);
  float _alpha = sqrt(1 + norm1 / norm0);
  alpha = complex<float>(-a.real() * _alpha, -a.imag() * _alpha);
  float rate = 1 + _alpha;
  a = complex<float>(a.real() * rate, a.imag() * rate);
  norm1 += complex_norm(a);
  norm1 = 1 / sqrt(norm1);
  a = complex<float>(a.real() * norm1, a.imag() * norm1);
  b = complex<float>(b.real() * norm1, b.imag() * norm1);
}

#define outer2(a, b, m0, m1) \
  do { \
    m0 = 1 - complex_norm(a) * 2; \
    m1 = complex<float>(complex_conj_mul(b, a)); \
    m1 = complex<float>(complex_mul_cons(m1, -2)); \
  } while(false)

/* Outer MM
 * (m0        m1) (a)
 * (m1.conj  -m0) (b)
 * */
#define lmm2x2(m0, m1, a, b) \
  do { \
    complex<float> \
      l0(complex_mul_cons(a, m0)), \
      r0(complex_mul(m1, b)), \
      l1(complex_conj_mul(m1, a)), \
      r1(complex_mul_cons(b, -m0)); \
    a = complex<float>(complex_add(l0, r0)); \
    b = complex<float>(complex_add(l1, r1)); \
  } while(false)

/*  Outer MM
 *  (a b) (m0       m1)
 *        (m1.conj -m0)
 * */
#define rmm2x2(a, b, m0, m1) \
  do { \
    complex<float> \
      l0(complex_mul_cons(a, m0)), \
      r0(complex_conj_mul(m1, b)), \
      l1(complex_mul(a, m1)), \
      r1(complex_mul_cons(b, -m0)); \
    a = complex<float>(complex_add(l0, r0)); \
    b = complex<float>(complex_add(l1, r1)); \
  } while(false)

void implicit_kernel(complex<float> *d, complex<float> *f, complex<float> *v, int n) {
  float mu = complex_norm(d[n - 1]);
  float m0;
  complex<float> m1;
  complex<float> a(complex_norm(d[0]) - mu), b(complex_conj_mul(f[0], d[0])), alpha;
  household2(a, b, alpha);
  outer2(a, b, m0, m1);
  for (int i = 0; i < N; ++i) {
    lmm2x2(m0, m1, v[i], v[i + N]);
  }
  a = d[0];
  b = complex<float>(0, 0);
  rmm2x2(a, f[0], m0, m1);
  rmm2x2(b, d[1], m0, m1);
  household2(a, b, d[0]);
  outer2(a, b, m0, m1);
  lmm2x2(m0, m1, f[0], d[1]);
  if (n != 2) {
    b = complex<float>(complex_mul(m1, f[1]));
    f[1] = complex<float>(complex_mul_cons(f[1], -m0));
  }
  for (int i = 1; i < n - 1; ++i) {
    a = std::conj(f[i - 1]);
    b = std::conj(b);
    household2(a, b, alpha);
    f[i - 1] = std::conj(alpha);
    outer2(a, b, m0, m1);
    a = d[i];
    rmm2x2(a, f[i], m0, m1);
    b = complex<float>(complex_conj_mul(m1, d[i + 1]));
    d[i + 1] = complex<float>(complex_mul_cons(d[i + 1], -m0));
    for (int j = 0; j < N; ++j) {
      lmm2x2(m0, m1, v[i * N + j], v[(i + 1) * N + j]);
    }
    household2(a, b, d[i]);
    outer2(a, b, m0, m1);
    lmm2x2(m0, m1, f[i], d[i + 1]);
    if (i != n - 2) {
      b = complex<float>(complex_mul(m1, f[i + 1]));
      f[i + 1] = complex<float>(complex_mul_cons(f[i + 1], -m0));
    }
  }
}

void svd(complex<float> *a, complex<float> *u, float *s, complex<float> *v) {
  for (int i = 0; i < N - 1; ++i) {
    int len = N - i;
    complex<float> hv[len], alpha;
    for (int j = 0; j < len; ++j)
      hv[j] = (i ? r : a)[j * len];
    household(hv, len, d[i]);

    for (int j = 1; j < len; ++j)
      temp[j] = 0;
    for (int k = 0; k < len; ++k) {
      for (int j = 1; j < len; ++j) {
        temp[j] += complex<float>(complex_conj_mul(hv[k], (i ? r : a)[k * len + j]));
      }
    }
    for (int j = 1; j < len; ++j)
      temp[j] = complex<float>(temp[j].real() * 2, temp[j].imag() * 2);
    //for (int j = 1; j < len; ++j) std::cout << temp[j] << " "; std::cout << "\n";

    for (int j = 0; j < len; ++j) {
      for (int k = 1; k < len; ++k) {
        complex<float> delta(complex_mul(temp[k], hv[j]));
        r[j * (len - 1) + (k - 1)] = complex<float>(complex_sub((i ? r : a)[j * len + k], delta));
      }
    }
    //for (int j = 0; j < len; ++j) { for (int k = 0; k < len - 1; ++k) std::cout << r[j * (len - 1) + k] << " "; std::cout << "\n"; }

    if (i != N - 2) {
      --len;
      for (int j = 0; j < len; ++j)
        hv[j] = r[j];
      household(hv, len, f[i]);

      for (int j = 0; j < len; ++j) {
        temp[j] = 0;
        for (int k = 0; k < len; ++k) {
          complex<float> delta(complex_conj_mul(hv[k], r[(j + 1) * len + k]));
          temp[j] = complex<float>(complex_add(temp[j], delta));
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
      } else {
        for (int k = 1; k < N; ++k)
          temp[k] = 0;
        for (int j = i + 1; j < N; ++j) {
          for (int k = 1; k < N; ++k) {
            complex<float> delta(complex_mul(hv[j - i - 1], v[j * N + k]));
            temp[k] = complex<float>(complex_add(temp[k], delta));
          }
        }
        for (int k = 1; k < N; ++k)
          temp[k] = complex<float>(temp[k].real() * 2, temp[k].imag() * 2);
        for (int k = 1; k < N; ++k) {
          for (int j = i + 1; j < N; ++j) {
            complex<float> delta(complex_conj_mul(hv[j - i - 1], temp[k]));
            v[j * N + k] -= delta;
          }
        }
      }
    }
  }
  f[N - 2] = r[0];
  d[N - 1] = r[1];
  for (bool next = true; next; ) {
    next = false;
    int i = 0;
    while (i < N - 1) {
      int j = i;
      while (j < N - 1 && (fabs(f[j].real()) > eps || fabs(f[j].imag()) > eps))
        ++j;
      if (i != j) {
        implicit_kernel(d + i, f + i, v + i * N, j - i + 1);
        next = true;
      }
      i = j + 1;
    }
  }
  for (int i = 0; i < N; ++i) {
    s[i] = sqrt(complex_norm(d[i]));
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex<float> sum(0, 0);
      for (int k = 0; k < N; ++k)
        sum += complex<float>(complex_conj_mul(v[j * N + k], a[i * N + k]));
      u[i * N + j] = sum;
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      u[i * N + j] /= s[j];
    }
  }
}
