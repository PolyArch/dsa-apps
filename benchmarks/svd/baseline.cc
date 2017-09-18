#include "svd.h"
#include <iostream>
#include <iomanip>
#include "sim_timing.h"

complex<float> _tmp[N * N];
const complex<float> _one(1, 0), _zero(0, 0);

void show_matrix(const char *name, complex<float> *a)  {
  std::cout << name << ":\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << a[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

float norm(complex<float> *a, int n) {
  float res = 0;
  for (int i = 0; i < n; ++i) {
    res += (a[i] * std::conj(a[i])).real();
  }
  return sqrt(res);
}

void household_vector(complex<float> *a, int y, int x, int n, complex<float> *v, complex<float> &w) {
  {
    for (int i = 0; i < n; ++i) {
      v[i] = a[(y + i) * N + x];
    }
    float sign = (v[0] * std::conj(v[0])).real();
    float _norm = norm(v, n);
    v[0] += std::exp(complex<float>(0, std::arg(v[0]))) * _norm; }
  {
    float _norm = norm(v, n);
    for (int i = 0; i < n; ++i) {
      v[i] /= _norm;
    }
  }
  {
    complex<float> xv(0, 0), vx(0, 0);
    for (int i = 0; i < n; ++i) {
      xv += std::conj(a[(y + i) * N + x]) * v[i];
      vx += std::conj(v[i]) * a[(y + i) * N + x]; }
    w = xv / vx + _one;
  }
}

void outer_mul_a(complex<float> *a, int y, int x, int n, int m, complex<float> *v, complex<float> w) {
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      _tmp[i * N + j] = complex<float>(0, 0);
      for (int k = 0; k < n; ++k) {
        _tmp[i * N + j] +=
          (((i - y) == k ? _one : _zero) - v[i - y] * std::conj(v[k]) * w) * a[(k + y) * N + j];
      }
    }
  }
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      a[i * N + j] = _tmp[i * N + j];
    }
  }
}

void a_mul_outer(complex<float> *a, int y, int x, int n, int m, complex<float> *v, complex<float> w) {
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      _tmp[i * N + j] = complex<float>(0, 0);
      for (int k = 0; k < m; ++k) {
        _tmp[i * N + j] +=
          a[i * N + k + x] * ((k == j - x ? _one : _zero) - v[k] * std::conj(v[j - x]) * w);
      }
    }
  }
  //show_matrix("tmp", _tmp);
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      a[i * N + j] = _tmp[i * N + j];
    }
  }
}

void hessenberg(complex<float> *a, complex<float> *h, complex<float> *inv) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      inv[i * N + j] = i == j ? _one : _zero;
      h[i * N + j] = a[i * N + j];
    }
  }
  for (int i = 1; i < N; ++i) {
    complex<float> v[N - i - 1], w;
    household_vector(h, i, i - 1, N - i, v, w);
    outer_mul_a(h, i, i - 1, N - i, N - i + 1, v, w);
    a_mul_outer(h, 0, i, N, N - i, v, w);
    a_mul_outer(inv, 0, i, N, N - i, v, w);
  }
}

void qr_hessenberg(complex<float> *a, complex<float> *q, complex<float> *r) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      q[i * N + j] = i == j ? _one : _zero;
      r[i * N + j] = a[i * N + j];
    }
  for (int i = 0; i < N - 1; ++i) {
    complex<float> v[2], w;
    household_vector(r, i, i, 2, v, w);
    outer_mul_a(r, i, i, 2, N - i, v, w);
    a_mul_outer(q, 0, i, N, 2, v, w);
  }
}

bool converged(complex<float> *a) {
  int cnt = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (fabs(a[i * N + j].real()) + fabs(a[i * N + j].imag()) < eps) {
        ++cnt;
      }
    }
  }
  return cnt == N * (N - 1);
}

void svd(complex<float> *a, complex<float> *u, complex<float> *s, complex<float> *v) {
  std::cout << std::setprecision(4);
  std::cout << std::fixed;

  complex<float> *at_a = new complex<float>[N * N];
  complex<float> *hes = new complex<float>[N * N];
  complex<float> *h_inv = new complex<float>[N * N];
  complex<float> *q = new complex<float>[N * N];
  complex<float> *r = new complex<float>[N * N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex<float> sum(0, 0);
      for (int k = 0; k < N; ++k)
        sum += std::conj(a[k * N + i]) * a[k * N + j];
      at_a[i * N + j] = sum;
    }
  }
  hessenberg(at_a, hes, h_inv);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      v[i * N + j] = i == j ? _one : _zero;
    }
  while (!converged(hes)) {
    qr_hessenberg(hes, q, r);
    for (int i = 0; i < N; ++i) {
      for (int j = i; j < N; ++j) {
        hes[i * N + j] = 0;
        for (int k = i; k < N; ++k) {
          hes[i * N + j] += r[i * N + k] * q[k * N + j];
        }
        hes[j * N + i] = std::conj(hes[i * N + j]);
      }
    }
    for (int i = 0; i < N * N; ++i)
      _tmp[i] = complex<float>(0, 0);
    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < N; ++k) {
        for (int j = k ? k - 1 : 0; j < N; ++j) {
          _tmp[i * N + j] += v[i * N + k] * q[k * N + j];
        }
      }
    }
    for (int i = 0; i < N * N; ++i)
      v[i] = _tmp[i];
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex<float> sum(0, 0);
      for (int k = 0; k < N; ++k) {
        sum += h_inv[i * N + k] * v[k * N + j];
      }
      _tmp[i * N + j] = sum;
    }
  }
  for (int i = 0; i < N * N; ++i)
    v[i] = _tmp[i];
  //show_matrix("inv", h_inv);
  //show_matrix("V", v);
  for (int i = 0; i < N; ++i) {
    s[i] = std::sqrt(hes[i * (N + 1)]);
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        u[i * N + j] += a[i * N + k] * v[k * N + j];
      }
    }
  }
  //show_matrix("U", u);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      u[j * N + i] /= s[i];
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      _tmp[j * N + i] = std::conj(v[i * N + j]);
    }
  }
  for (int i = 0; i < N * N; ++i) {
    v[i] = _tmp[i];
  }
}

