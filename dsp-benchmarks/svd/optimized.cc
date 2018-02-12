#include "svd.h"
#include "matvec.h"

static complex<float> f[_N_], d[_N_], r[_N_ * _N_], temp[_N_], another[_N_];

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
  for (int i = 0; i < _N_; ++i) {
    lmm2x2(m0, m1, v[i], v[i + _N_]);
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
    for (int j = 0; j < _N_; ++j) {
      lmm2x2(m0, m1, v[i * _N_ + j], v[(i + 1) * _N_ + j]);
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
  for (int i = 0; i < _N_ - 1; ++i) {
    int len = _N_ - i;
    complex<float> hv[len], alpha;
    for (int j = 0; j < len; ++j)
      hv[j] = (i ? r : a)[j * len];
    household(hv, len, d[i]);

    CPUvec_mul_mat((i ? r : a) + 1, len, len, len, hv, true, temp + 1);
    //CPUsub_outerx2((i ? r : a) + 1, len, len - 1, len, hv, temp + 1, false, r, len - 1);
    for (int j = 0; j < len; ++j) {
      for (int k = 1; k < len; ++k) {
        complex<float> delta(temp[k] * hv[j]);
        delta *= 2;
        r[j * (len - 1) + (k - 1)] = complex<float>(complex_sub((i ? r : a)[j * len + k], delta));
      }
    }

    if (i != _N_ - 2) {
      --len;
      for (int j = 0; j < len; ++j)
        hv[j] = r[j];
      household(hv, len, f[i]);

      CPUmat_mul_vec(r + len, len, len, len, hv, true, temp);
      CPUsub_outerx2(r + len, len, len, len, temp, hv, false, r, len);

      if (!i) {
        v[0] = complex<float>(1, 0);
        for (int j = 1; j < _N_; ++j) {
          v[j] = v[j * _N_] = complex<float>(0, 0);
          for (int k = 1; k < _N_; ++k) {
            complex<float> delta(complex_conj_mul(hv[j - 1], hv[k - 1]));
            complex<float> diag(j == k, 0);
            complex<float> val(delta.real() * 2, delta.imag() * 2);
            v[j * _N_ + k] = complex<float>(complex_sub(diag, val));
          }
        }
      } else {
        CPUvec_mul_mat(v + (i + 1) * _N_ + 1, len, _N_ - 1, _N_, hv, false, temp + 1);
        //CPUsub_outerx2(v + (i + 1) * _N_ + 1, len, _N_ - 1, _N_, temp + 1, hv, true, v + (i + 1) * _N_ + 1, _N_);
        for (int j = i + 1; j < _N_; ++j) {
          for (int k = 1; k < _N_; ++k) {
            complex<float> delta(std::conj(hv[j - i - 1]) * temp[k]);
            delta *= 2;
            v[j * _N_ + k] -= delta;
          }
        }
      }
    }
  }
  f[_N_ - 2] = r[0];
  d[_N_ - 1] = r[1];
  for (bool next = true; next; ) {
    next = false;
    int i = 0;
    while (i < _N_ - 1) {
      int j = i;
      while (j < _N_ - 1 && (fabs(f[j].real()) > eps || fabs(f[j].imag()) > eps))
        ++j;
      if (i != j) {
        implicit_kernel(d + i, f + i, v + i * _N_, j - i + 1);
        next = true;
      }
      i = j + 1;
    }
  }
  for (int i = 0; i < _N_; ++i) {
    s[i] = sqrt(complex_norm(d[i]));
  }
  for (int i = 0; i < _N_; ++i) {
    for (int j = 0; j < _N_; ++j) {
      complex<float> sum(0, 0);
      for (int k = 0; k < _N_; ++k)
        sum += complex<float>(complex_conj_mul(v[j * _N_ + k], a[i * _N_ + k]));
      u[i * _N_ + j] = sum;
    }
  }
  for (int i = 0; i < _N_; ++i) {
    for (int j = 0; j < _N_; ++j) {
      u[i * _N_ + j] /= s[j];
    }
  }
}
