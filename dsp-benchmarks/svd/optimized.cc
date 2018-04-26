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

void givens(complex<float> &a, complex<float> &b, complex<float> &alpha) {
  float norm0 = complex_norm(a), norm1 = complex_norm(b);
  float r = sqrt(norm1 + norm0);
  complex<float> c(std::conj(a) / r);
  complex<float> s(std::conj(b) / r);
  alpha = r;
  a = c;
  b = s;
}

/* L Givens Rotation
 * (c        s     ) (a)
 * (s.conj  -c.conj) (b)
 * */
#define apply_givens(c, s, a, b) \
  do { \
    complex<float> \
      resa((c) * (a) + (s) * (b)), \
      resb(std::conj(s) * (a) - std::conj(c) * (b)); \
    a = resa; \
    b = resb; \
  } while(false)

void implicit_kernel(complex<float> *d, complex<float> *f, complex<float> *v, int n) {
  float mu = complex_norm(d[n - 1]);
  complex<float> a(complex_norm(d[0]) - mu), b(complex_conj_mul(f[0], d[0])), alpha;
  complex<float> c, s, extra;
  givens(a, b, alpha);

  c = a; s = b;
  for (int i = 0; i < _N_; ++i) {
    apply_givens(c, s, v[i], v[i + _N_]);
  }

  a = d[0] * c + f[0] * std::conj(s);
  f[0] = d[0] * s - f[0] * std::conj(c);
  b = d[1] * std::conj(s);
  d[1] *= -std::conj(c);

  givens(a, b, d[0]);
  apply_givens(a, b, f[0], d[1]);
  if (n != 2) {
    b *= f[1];
    f[1] *= -std::conj(a);
  }

  for (int i = 1; i < n - 1; ++i) {
    a = f[i - 1];
    givens(a, b, f[i - 1]);
    c = a; s = b;
    a = d[i];

    a = d[i] * c + f[i] * s;
    f[i] = d[i] * std::conj(s) - f[i] * std::conj(c);
    b = d[i + 1] * s;
    d[i + 1] *= -std::conj(c);
    for (int j = 0; j < _N_; ++j) {
      apply_givens(c, std::conj(s), v[i * _N_ + j], v[(i + 1) * _N_ + j]);
    }

    givens(a, b, d[i]);
    c = a; s = b;
    apply_givens(c, s, f[i], d[i + 1]);

    if (i != n - 2) {
      b = s * f[i + 1];
      f[i + 1] *= -std::conj(c);
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

  int left = 0, right = _N_ - 1;
  while (left < right) {
    while (left < _N_ - 1 && fabs(f[left].real()) < eps && fabs(f[left].imag()) < eps)
      ++left;
    while (right >= 1 && fabs(f[right - 1].real()) < eps && fabs(f[right - 1].imag()) < eps)
      --right;
    if (right - left >= 1) {
      //std::cout << left << " " << right << "\n";
      implicit_kernel(d + left, f + left, v + left * _N_, right - left + 1);
      //for (int i = left; i < right; ++i) std::cout << f[i] << " "; std::cout << "\n";
      //for (int i = left; i < right + 1; ++i) std::cout << d[i] << " "; std::cout << "\n";
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
