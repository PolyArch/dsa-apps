#include "svd.h"
#include <iostream>
#include <iomanip>
#include "sim_timing.h"
#include "sb_insts.h"
#include "hes_mul.h"
#include "mm_sb.h"
#include "mul_acc.h"
#include "conj_mul_acc.h"
#include "fin_a_mul_outer.h"
#include "fin_outer_mul_a.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

#define sb_gemm(r, q, hes) \
            SB_DMA_READ(r + i_row + kk, 8, 8, BN, P_mm_sb_A); \
            SB_DMA_READ(q + k_row + jj, NN * 8, BN * 8, BN, P_mm_sb_B); \
            SB_CONST(P_mm_sb_A, *((uint64_t*)&_one), 1); \
            SB_DMA_READ(hes + i_row + jj, 0, BN * 8, 1, P_mm_sb_B); \
            SB_CONST(P_mm_sb_reset, 0, BN); \
            SB_CONST(P_mm_sb_reset, 1, 1);\
            SB_GARBAGE(P_mm_sb_R, BN * BN); \
            SB_DMA_WRITE(P_mm_sb_R, 8, 8, BN, hes + i_row + jj);

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
  SB_CONFIG(conj_mul_acc_config, conj_mul_acc_size);
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      //_tmp[i * NN + j] = complex<float>(0, 0);
      complex<float> sum(0, 0), *vk = v, *ak = a + y * NN + j;
      SB_DMA_READ(vk, 8, 8, n, P_conj_mul_acc_A);
      SB_DMA_READ(ak, 8 * NN, 8, n, P_conj_mul_acc_B);
      SB_CONST(P_conj_mul_acc_reset, 0, n - 1);
      SB_CONST(P_conj_mul_acc_reset, 1, 1);
      SB_GARBAGE(P_conj_mul_acc_R, n - 1);
      SB_DMA_WRITE(P_conj_mul_acc_R, 8, 8, 1, &_tmp[i * NN + j]);
      /*for (int k = 0; k < n; ++k) {
        //sum += std::conj(v[k]) * a[(k + y) * N + j];
        sum += complex<float>(complex_conj_mul(*vk, *ak)); //delta;
        ++vk;
        ak += NN;
      }
      _tmp[i * NN + j] = sum;*/
    }
  }
  SB_WAIT_ALL();

  SB_CONFIG(fin_outer_mul_a_config, fin_outer_mul_a_size);
  complex<float> *vi = v;
  w = -w;
  for (int i = y; i < y + n; ++i) {
    SB_CONST(P_fin_outer_mul_a_VI, *((uint64_t *)(vi++)), m);
    SB_CONST(P_fin_outer_mul_a_W, *((uint64_t *)&w), m);
    SB_DMA_READ(&_tmp[i * NN + x], 8, 8, m, P_fin_outer_mul_a_TMP);
    SB_DMA_READ(&a[i * NN + x], 8, 8, m, P_fin_outer_mul_a_A);
    SB_DMA_WRITE(P_fin_outer_mul_a_R, 8, 8, m, &a[i * NN + x]);

    /*
    for (int j = x; j < x + m; ++j) {
      complex<float> val(complex_mul(*vi, w));
      a[i * NN + j] -= complex<float>(complex_mul(val, _tmp[i * NN + j]));
    }
    ++vi;*/
  }
  SB_WAIT_ALL();
}

void a_mul_outer(complex<float> *a, int y, int x, int n, int m, complex<float> *v, complex<float> w) {

  SB_CONFIG(mul_acc_config, mul_acc_size);
  for (int i = y; i < y + n; ++i) {
    for (int j = x; j < x + m; ++j) {
      complex<float> sum(0, 0), *ak = a + i * NN + x, *vk = v;
      SB_DMA_READ(ak, 8, 8, m, P_mul_acc_A);
      SB_DMA_READ(vk, 8, 8, m, P_mul_acc_B);
      SB_CONST(P_mul_acc_reset, 0, m - 1);
      SB_CONST(P_mul_acc_reset, 1, 1);
      SB_GARBAGE(P_mul_acc_R, m - 1);
      SB_DMA_WRITE(P_mul_acc_R, 8, 8, 1, &_tmp[i * NN + j]);
      /*for (int k = 0; k < m; ++k) {
        sum += complex<float>(complex_mul(*ak, *vk));
        ++ak;
        ++vk;
      }
      _tmp[i * NN + j] = sum;*/
    }
  }
  SB_WAIT_ALL();

  w = -w;
  SB_CONFIG(fin_a_mul_outer_config, fin_a_mul_outer_size);
  for (int i = y; i < y + n; ++i) {
    complex<float> *tmpj = _tmp + i * NN + x, *vj = v;

    SB_DMA_READ(tmpj, 8, 8, m, P_fin_a_mul_outer_TMPJ);
    SB_CONST(P_fin_a_mul_outer_W, *((uint64_t *) &w), m);
    SB_DMA_READ(vj, 8, 8, m, P_fin_a_mul_outer_VJ);
    SB_DMA_READ(&a[i * NN + x], 8, 8, m, P_fin_a_mul_outer_A);
    SB_DMA_WRITE(P_fin_a_mul_outer_R, 8, 8, m, &a[i * NN + x]);
    /*for (int j = x; j < x + m; ++j) {
      complex<float> val(complex_mul(*tmpj, w));
      complex<float> delta(complex_conj_mul(*vj, val));
      a[i * NN + j] -= delta;
      ++tmpj;
      ++vj;
    }*/
  }
  SB_WAIT_ALL();
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

    SB_CONFIG(hes_mul_config, hes_mul_size);
    SB_CONST(P_hes_mul__H0, *((uint64_t *)(&h0)), N - i);
    SB_CONST(P_hes_mul__H1, *((uint64_t *)(&h1)), N - i);
    SB_CONST(P_hes_mul__H2, *((uint64_t *)(&h2)), N - i);
    SB_CONST(P_hes_mul__H3, *((uint64_t *)(&h3)), N - i);
    SB_DMA_READ(r0, 8, 8, N - i, P_hes_mul__R0);
    SB_DMA_READ(r1, 8, 8, N - i, P_hes_mul__R1);
    SB_DMA_WRITE(P_hes_mul_X, 8, 8, N - i, r0);
    SB_DMA_WRITE(P_hes_mul_Y, 8, 8, N - i, r1);

    /*for (int j = i; j < N; ++j) {
      complex<float> lx(complex_mul(h0, *r0));
      complex<float> rx(complex_mul(h1, *r1));
      complex<float> ly(complex_mul(h2, *r0));
      complex<float> ry(complex_mul(h3, *r1));
      r[i * NN + j] = complex<float>(complex_add(lx, rx));
      r[(i + 1) * NN + j] = complex<float>(complex_add(ly, ry));
      ++r0;
      ++r1;
    }*/

    complex<float> *q0 = q + i, *q1 = q + i + 1;

    SB_CONST(P_hes_mul__H0, *((uint64_t *)(&h0)), N);
    SB_CONST(P_hes_mul__H1, *((uint64_t *)(&h2)), N);
    SB_CONST(P_hes_mul__H2, *((uint64_t *)(&h1)), N);
    SB_CONST(P_hes_mul__H3, *((uint64_t *)(&h3)), N);
    SB_DMA_READ(q0, 8 * NN, 8, N, P_hes_mul__R0);
    SB_DMA_READ(q1, 8 * NN, 8, N, P_hes_mul__R1);
    SB_DMA_WRITE(P_hes_mul_X, 8 * NN, 8, N, q0);
    SB_DMA_WRITE(P_hes_mul_Y, 8 * NN, 8, N, q1);

    /*for (int j = 0; j < N; ++j) {
      complex<float> lx(complex_mul(h0, *q0));
      complex<float> rx(complex_mul(h2, *q1));
      complex<float> ly(complex_mul(h1, *q0)), ry(complex_mul(h3, *q1));
      q[j * NN + i] = complex<float>(complex_add(lx, rx));
      q[j * NN + i + 1] = complex<float>(complex_add(ly, ry));
      q0 += NN;
      q1 += NN;
    }*/
  }
  SB_WAIT_ALL();
}

bool converged(complex<float> *a) {
  int cnt = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (fabs(a[i * NN + j].real()) + fabs(a[i * NN + j].imag()) < eps) {
        ++cnt;
      }
    }
  }
  return cnt == N * (N - 1);
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
      SB_CONFIG(mm_sb_config, mm_sb_size);
      for (int jj = 0; jj < N; jj += BN){
        for (int kk = 0; kk < N; kk += BN){
          int _i = N;
          _i = _i > jj + BN ? jj + BN : _i;
          _i = _i > kk + BN ? kk + BN : _i;
          for (int i = 0; i < _i; ++i) {
            /*if (jj + BN - 1 < i || kk + BN - 1 < i) {
              break;
            }*/
            int i_row = i * NN;
            int k_row = kk * NN;

            /*for (int k = 0; k < BN; ++k) {
              SB_CONST(P_mm_sb_A, *((uint64_t*)&r[i_row + k + kk]), 4);
            }*/

            sb_gemm(r, q, hes)

            /*SB_DMA_READ(r + i_row + kk, 8, 8, BN, P_mm_sb_A);
            SB_DMA_READ(q + k_row + jj, NN * 8, BN * 8, BN, P_mm_sb_B);
            SB_CONST(P_mm_sb_A, *((uint64_t*)&_one), 1);
            SB_DMA_READ(hes + i_row + jj, 0, BN * 8, 1, P_mm_sb_B);
            SB_CONST(P_mm_sb_reset, 0, BN);
            SB_CONST(P_mm_sb_reset, 1, 1);
            SB_GARBAGE(P_mm_sb_R, BN * BN);
            SB_DMA_WRITE(P_mm_sb_R, 8, 8, BN, hes + i_row + jj);*/

            /*for (int k = 0; k < BN; ++k) {
              int i_row = i * NN;
              int k_row = (k + kk) * NN;
              complex<float> temp_x = (r)[i_row + k + kk];
              for (int j = 0; j < BN; ++j) {
                complex<float> mul(complex_mul(temp_x, (q)[k_row + j + jj]));
                hes[i_row + j + jj] += mul;
              }
            }*/
          }
        }
      }
      //SB_WAIT_ALL();
      /*for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
          hes[j * NN + i] = std::conj(hes[i * NN + j]);
        }
      }*/

    }
    
    /*complex<float> ref[NN * NN];
    for (int i = 0; i < N; ++i) {
      for (int j = i; j < N; ++j) {
        ref[i * NN + j] = 0;
        for (int k = i; k < N; ++k) {
          complex<float> delta(complex_mul(r[i * NN + k], q[k * NN + j]));
          ref[i * NN + j] += delta;
        }
        std::cout << "ref: " << ref[i * NN + j] << ", hes: " << hes[i * NN + j] << "\n";
        assert(fabs(ref[i * NN + j].real() - hes[i * NN + j].real()) < eps * 10);
        assert(fabs(ref[i * NN + j].imag() - hes[i * NN + j].imag()) < eps * 10);
        //hes[j * NN + i] = std::conj(hes[i * NN + j]);
      }
    }*/
    
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        _tmp[i * NN + j] = complex<float>(0, 0);

    {
      for (int jj = 0; jj < N; jj += BN){
        for (int kk = 0; kk < N; kk += BN){
          /*if (jj + BN - 1 < kk - 1) {
            break;
          }*/
          for (int i = 0; i < N; ++i){
            int i_row = i * NN;
            int k_row = kk * NN;

            sb_gemm(vv, q, _tmp)
            
            /*for (int k = 0; k < BN; ++k){
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
            }*/
          }
        }
      }

      SB_WAIT_ALL();

      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          vv[i * NN + j] = _tmp[i * NN + j];
      for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
          hes[j * NN + i] = std::conj(hes[i * NN + j]);
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
  }
  //show_matrix("V", vv);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      _tmp[i * NN + j] = complex<float>(0, 0);

  {
    SB_CONFIG(mm_sb_config, mm_sb_size);
    for (int jj = 0; jj < N; jj += BN){
      for (int kk = 0; kk < N; kk += BN){
        for (int i = 0; i < N; ++i){
          int i_row = i * NN;
          int k_row = kk * NN;

          sb_gemm(h_inv, vv, _tmp)
          /*for (int k = 0; k < BN; ++k){
            complex<float> temp_x = (h_inv)[i_row + k + kk];
            for (int j = 0; j < BN; ++j){
              complex<float> mul(complex_mul(temp_x, (vv)[k_row + j + jj]));
              _tmp[i_row + j + jj] += mul;
            }
          }*/
        }
      }
    }
    SB_WAIT_ALL();
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

