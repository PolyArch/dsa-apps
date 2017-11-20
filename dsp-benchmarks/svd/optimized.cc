#include "svd.h"
#include <iostream>
#include <iomanip>
#include "sim_timing.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

/*
    loopjj:for (jj = 0; jj < N; jj += block_size){
        loopkk:for (kk = 0; kk < N; kk += block_size){
            loopi:for ( i = 0; i < N; ++i){
                loopk:for (k = 0; k < block_size; ++k){
                    i_row = i * NN;
                    k_row = (k + kk) * NN;
                    temp_x = (m1)[i_row + k + kk];
                    loopj:for (j = 0; j < block_size; ++j){
                        mul = temp_x * (m2)[k_row + j + jj];
                        prod[i_row + j + jj] += mul;
                    }
                }
            }
        }
    }
*/

complex<float> _tmp[NN * NN];
const complex<float> _one(1, 0), _zero(0, 0);

void show_matrix(const char *name, complex<float> *a)  {
  std::cout << name << ":\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << a[i * NN + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

float norm(complex<float> *a, int n) {
  float res = 0;
  for (int i = 0; i < n; ++i) {
    res += complex_norm(a[i]);
  }
  return sqrt(res);
}

void household_vector(complex<float> *a, int y, int x, int n, complex<float> *v, complex<float> &w) {
  {
    for (int i = 0; i < n; ++i) {
      v[i] = a[(y + i) * NN + x];
    }
    float sign = sqrt(complex_norm(*v));
    float _norm = norm(v, n);
    v[0] += complex<float>(v[0].real() / sign * _norm, v[0].imag() / sign * _norm);
    //v[0] += std::exp(complex<float>(0, std::arg(v[0]))) * _norm;
  }
  {
    float _norm = norm(v, n);
    for (int i = 0; i < n; ++i) {
      v[i] /= _norm;
    }
  }
  {
    complex<float> xv(0, 0), vx(0, 0);
    complex<float> *cur = a + y * NN + x;
    for (int i = 0; i < n; ++i) {
      //xv += std::conj(*cur) * v[i];
      //vx += std::conj(v[i]) * (*cur);
      xv += complex<float>(complex_conj_mul(*cur, v[i]));
      vx += complex<float>(complex_conj_mul(v[i], *cur));
      cur += NN;
    }
    //w = xv / vx + _one;
    float norm = vx.real() * vx.real() + vx.imag() * vx.imag();
    w = complex<float>(
        (xv.real() * vx.real() + xv.imag() * vx.imag()) / norm + 1,
        (xv.imag() * vx.real() - xv.real() * vx.imag()) / norm);
  }
}

void outer_mul_a(complex<float> *a, int y, int x, int n, int m, complex<float> *v, complex<float> w) {
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      _tmp[i * NN + j] = complex<float>(0, 0);
      complex<float> sum(0, 0), *vk = v, *ak = a + y * NN + j;
      for (int k = 0; k < n; ++k) {
        //sum += std::conj(v[k]) * a[(k + y) * N + j];
        sum += complex<float>(complex_conj_mul(*vk, *ak)); //delta;
        ++vk;
        ak += NN;
      }
      _tmp[i * NN + j] = sum;
    }
  }
  complex<float> *vi = v;
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      complex<float> val(complex_mul(*vi, w));
      a[i * NN + j] -= complex<float>(complex_mul(val, _tmp[i * NN + j]));
    }
    ++vi;
  }
}

void a_mul_outer(complex<float> *a, int y, int x, int n, int m, complex<float> *v, complex<float> w) {
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      complex<float> sum(0, 0), *ak = a + i * NN + x, *vk = v;
      for (int k = 0; k < m; ++k) {
        sum += complex<float>(complex_mul(*ak, *vk));
        //sum += a[i * N + k + x] * v[k];
        ++ak;
        ++vk;
      }
      _tmp[i * NN + j] = sum;
    }
  }
  for (int i = y; i < y + n; ++i) {
    complex<float> *tmpj = _tmp + i * NN + x, *vj = v;
    for (int j = x; j < x + m; ++j) {
      complex<float> val(complex_mul(*tmpj, w));
      complex<float> delta(complex_conj_mul(*vj, val));
      a[i * NN + j] -= delta;
      ++tmpj;
      ++vj;
    }
  }
}

void hessenberg(complex<float> *a, complex<float> *h, complex<float> *inv) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      inv[i * NN + j] = i == j ? _one : _zero;
      h[i * NN + j] = a[i * NN + j];
    }
  }
  for (int i = 1; i < N; ++i) {
    complex<float> v[N - i - 1 + BN], w;
    household_vector(h, i, i - 1, N - i, v, w);
    outer_mul_a(h, i, i - 1, N - i, N - i + 1, v, w);
    a_mul_outer(h, 0, i, N, N - i, v, w);
    a_mul_outer(inv, 0, i, N, N - i, v, w);
  }
}

void qr_hessenberg(complex<float> *a, complex<float> *q, complex<float> *r) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      q[i * NN + j] = i == j ? _one : _zero;
      r[i * NN + j] = a[i * NN + j];
    }
  for (int i = 0; i < N - 1; ++i) {
    complex<float> v[2], w, h0, h1, h2, h3;
    //household_vector(r, i, i, 2, v, w);
    {
      v[0] = r[i * NN + i];
      v[1] = r[(i + 1) * NN + i];
      float sign = sqrt(complex_norm(v[0]));
      float _norm = sqrt(complex_norm(v[0]) + complex_norm(v[1]));
      v[0] += complex<float>(v[0].real() / sign * _norm, v[0].imag() / sign * _norm);
      _norm = sqrt(complex_norm(v[0]) + complex_norm(v[1]));
      v[0] /= _norm;
      v[1] /= _norm;
      complex<float> xv(0, 0), vx(0, 0);
      xv += complex<float>(complex_conj_mul(r[i * NN + i], v[0]));
      vx += complex<float>(complex_conj_mul(v[0], r[i * NN + i]));
      xv += complex<float>(complex_conj_mul(r[(i + 1) * NN + i], v[1]));
      vx += complex<float>(complex_conj_mul(v[1], r[(i + 1) * NN + i]));
      _norm = vx.real() * vx.real() + vx.imag() * vx.imag();
      w = complex<float>(
        (xv.real() * vx.real() + xv.imag() * vx.imag()) / _norm + 1,
        (xv.imag() * vx.real() - xv.real() * vx.imag()) / _norm);

    }

    //h[0] = _one - w * v[0] * std::conj(v[0]);
    float v_00 = complex_norm(v[0]);
    h0 = complex<float>(1 - w.real() * v_00, -w.imag() * v_00);
    complex<float> val;
    val = complex<float>(complex_conj_mul(v[1], v[0]));
    //h[1] = -w * v[0] * std::conj(v[1]);
    h1 = complex<float>(complex_mul(-w, val));
    //h[2] = -w * v[1] * std::conj(v[0]);
    val = complex<float>(complex_conj_mul(v[0], v[1]));
    h2 = complex<float>(complex_mul(-w, val));
    float v_11 = complex_norm(v[1]);
    //h[3] = _one - w * v[1] * std::conj(v[1]);
    h3 = complex<float>(1 - w.real() * v_11, -w.imag() * v_11);

    complex<float> *r0 = r + i * NN + i;
    complex<float> *r1 = r + (i + 1) * NN + i;
    for (int j = i; j < N; ++j) {
      complex<float> lx(complex_mul(h0, *r0));
      complex<float> rx(complex_mul(h1, *r1));
      complex<float> ly(complex_mul(h2, *r0));
      complex<float> ry(complex_mul(h3, *r1));
      r[i * NN + j] = complex<float>(complex_add(lx, rx));
      r[(i + 1) * NN + j] = complex<float>(complex_add(ly, ry));
      ++r0;
      ++r1;
    }

    complex<float> *q0 = q + i, *q1 = q + i + 1;
    for (int j = 0; j < N; ++j) {
      complex<float> lx(complex_mul(h0, *q0));
      complex<float> rx(complex_mul(h2, *q1));
      complex<float> ly(complex_mul(h1, *q0)), ry(complex_mul(h3, *q1));
      q[j * NN + i] = complex<float>(complex_add(lx, rx));
      q[j * NN + i + 1] = complex<float>(complex_add(ly, ry));
      q0 += NN;
      q1 += NN;
    }
  }
}

bool converged(complex<float> *a) {
  static int last = -1, keep = 0;
  int cnt = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (fabs(a[i * NN + j].real()) > eps || fabs(a[i * NN + j].imag()) > eps) {
        ++cnt;
      }
    }
  }
  if (last != cnt) {
    last = cnt;
    keep = 1;
  } else
    ++keep;

  return cnt == N || keep > 50;
}

void svd(complex<float> *a, complex<float> *u, complex<float> *s, complex<float> *v) {
  std::cout << std::setprecision(6);
  std::cout << std::fixed;

  complex<float> *at_a = new complex<float>[NN * NN];
  complex<float> *hes = new complex<float>[NN * NN];
  complex<float> *h_inv = new complex<float>[NN * NN];
  complex<float> *q = new complex<float>[NN * NN];
  complex<float> *r = new complex<float>[NN * NN];
  complex<float> *vv = new complex<float>[NN * NN];

  for (int i = 0; i < NN * NN; ++i) {
    at_a[i] = _zero;
  }
  for (int k = 0; k < N; ++k)
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        complex<float> delta(
            complex_conj_mul(a[k * N + i], a[k * N + j])
        );
        at_a[i * NN + j] += delta;
      }
    }

  hessenberg(at_a, hes, h_inv);
  //show_matrix("hes", hes);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      vv[i * NN + j] = i == j ? _one : _zero;
    }
  while (!converged(hes)) {
    qr_hessenberg(hes, q, r);
    {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          hes[i * NN + j] = _zero;
      for (int jj = 0; jj < N; jj += BN){
        for (int kk = 0; kk < N; kk += BN){
          for (int i = 0; i < N; ++i) {
            if (jj + BN - 1 < i || kk + BN - 1 < i) {
              break;
            }
            for (int k = 0; k < BN; ++k) {
              int i_row = i * NN;
              int k_row = (k + kk) * NN;
              complex<float> temp_x = (r)[i_row + k + kk];
              for (int j = 0; j < BN; ++j) {
                complex<float> mul(complex_mul(temp_x, (q)[k_row + j + jj]));
                hes[i_row + j + jj] += mul;
              }
            }
          }
        }
      }
      for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
          hes[j * NN + i] = std::conj(hes[i * NN + j]);
        }
      }
    }
    /*
    for (int i = 0; i < N; ++i) {
      for (int j = i; j < N; ++j) {
        hes[i * NN + j] = 0;
        for (int k = i; k < N; ++k) {
          complex<float> delta(complex_mul(r[i * NN + k], q[k * NN + j]));
          hes[i * NN + j] += delta;
        }
        hes[j * NN + i] = std::conj(hes[i * NN + j]);
      }
    }
    */
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        _tmp[i * NN + j] = complex<float>(0, 0);

    {
      for (int jj = 0; jj < N; jj += BN){
        for (int kk = 0; kk < N; kk += BN){
          if (jj + BN - 1 < kk - 1) {
            break;
          }
          for (int i = 0; i < N; ++i){
            for (int k = 0; k < BN; ++k){
              int _k = k + kk;
              if (jj + BN - 1 < _k - 1) {
                break;
              }
              int i_row = i * NN;
              int k_row = (k + kk) * NN;
              complex<float >temp_x = (vv)[i_row + k + kk];
              for (int j = 0; j < BN; ++j){
                complex<float> mul(complex_mul(temp_x, (q)[k_row + j + jj]));
                _tmp[i_row + j + jj] += mul;
              }
            }
          }
        }
      }
    }

    /*for (int i = 0; i < N; ++i) {
      for (int k = 0; k < N; ++k) {
        for (int j = k ? k - 1 : 0; j < N; ++j) {
          _tmp[i * NN + j] += complex<float>(complex_mul(vv[i * NN + k], q[k * NN + j]));
        }
      }
    }*/

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        vv[i * NN + j] = _tmp[i * NN + j];
  }
  //show_matrix("V", vv);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      _tmp[i * NN + j] = complex<float>(0, 0);
  {
    for (int jj = 0; jj < N; jj += BN){
      for (int kk = 0; kk < N; kk += BN){
        for (int i = 0; i < N; ++i){
          for (int k = 0; k < BN; ++k){
            int i_row = i * NN;
            int k_row = (k + kk) * NN;
            complex<float> temp_x = (h_inv)[i_row + k + kk];
            for (int j = 0; j < BN; ++j){
              complex<float> mul(complex_mul(temp_x, (vv)[k_row + j + jj]));
              _tmp[i_row + j + jj] += mul;
            }
          }
        }
      }
    }
  }
/*
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex<float> sum(0, 0);
      for (int k = 0; k < N; ++k) {
        sum += complex<float>(complex_mul(h_inv[i * NN + k], vv[k * NN + j]));
      }
      _tmp[i * NN + j] = sum;
    }
  }
*/
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      vv[i * NN + j] = _tmp[i * NN + j];
  //show_matrix("V", vv);
  for (int i = 0; i < N; ++i) {
    s[i] = std::sqrt(hes[i * (NN + 1)]);
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex<float> sum(0, 0);
      for (int k = 0; k < N; ++k) {
        sum += complex<float>(complex_mul(a[i * N + k], vv[k * NN + j]));
      }
      u[i * N + j] = sum;
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      u[j * N + i] /= s[i];
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      _tmp[j * NN + i] = std::conj(vv[i * NN + j]);
    }
  }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      v[i * N + j] = _tmp[i * NN + j];
}

