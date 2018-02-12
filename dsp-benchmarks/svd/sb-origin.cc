#include "svd.h"
#include "sb_insts.h"
#include "vmc.dfg.h"
#include "vv.dfg.h"
#include "mvc.dfg.h"
#include "vvc.dfg.h"
#include "vm.dfg.h"
#include "lmm2x2.dfg.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_mul_cons(a, b) (a).real() * (b), (a).imag() * (b)

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

complex<float> f[_N_], d[_N_], r[_N_ * _N_], temp[_N_];
complex<float> _one(1, 0), _zero(0, 0);

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
  union _reiter_t {
    float a[2];
    uint64_t val;
  } ri_m0 = {m0, m0};
  SB_CONFIG(lmm2x2_config, lmm2x2_size);
  SB_CONST(P_lmm2x2_M0, ri_m0.val, _N_);
  SB_CONST(P_lmm2x2_M1, *((uint64_t*)&m1), _N_);
  SB_DMA_READ(v, 8, 8, _N_, P_lmm2x2_A);
  SB_DMA_READ(v + _N_, 8, 8, _N_, P_lmm2x2_B);
  SB_DMA_WRITE(P_lmm2x2_O0, 8, 8, _N_, v);
  SB_DMA_WRITE(P_lmm2x2_O1, 8, 8, _N_, v + _N_);
  SB_WAIT_ALL();
  //for (int i = 0; i < _N_; ++i) {
  //  lmm2x2(m0, m1, v[i], v[i + _N_]);
  //}
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
    ri_m0 = (_reiter_t) {m0, m0};
    SB_CONST(P_lmm2x2_M0, ri_m0.val, _N_);
    SB_CONST(P_lmm2x2_M1, *((uint64_t*)&m1), _N_);
    SB_DMA_READ(v + i * _N_, 8, 8, _N_, P_lmm2x2_A);
    SB_DMA_READ(v + i * _N_ + _N_, 8, 8, _N_, P_lmm2x2_B);
    SB_DMA_WRITE(P_lmm2x2_O0, 8, 8, _N_, v + i * _N_);
    SB_DMA_WRITE(P_lmm2x2_O1, 8, 8, _N_, v + i * _N_ + _N_);
    //for (int j = 0; j < _N_; ++j) {
    //  lmm2x2(m0, m1, v[i * _N_ + j], v[(i + 1) * _N_ + j]);
    //}
    household2(a, b, d[i]);
    outer2(a, b, m0, m1);
    lmm2x2(m0, m1, f[i], d[i + 1]);
    if (i != n - 2) {
      b = complex<float>(complex_mul(m1, f[i + 1]));
      f[i + 1] = complex<float>(complex_mul_cons(f[i + 1], -m0));
    }
    SB_WAIT_ALL();
  }
}

void svd(complex<float> *a, complex<float> *u, float *s, complex<float> *v) {
  for (int i = 0; i < _N_ - 1; ++i) {
    int len = _N_ - i;
    complex<float> hv[len], alpha;
    for (int j = 0; j < len; ++j)
      hv[j] = (i ? r : a)[j * len];
    household(hv, len, d[i]);

    SB_CONFIG(vmc_config, vmc_size);
    SB_CONST(P_vmc_C, *((uint64_t*)&_zero), len - 1);
    SB_DMA_READ((i ? r : a) + 1, 8 * len, 8 * (len - 1), len, P_vmc_B);
    SB_RECURRENCE(P_vmc_O, P_vmc_C, (len - 1) * (len - 1));
    for (int k = 0; k < len; ++k) {
      SB_CONST(P_vmc_A, *((uint64_t*)(hv + k)), len - 1);
    }
    SB_DMA_WRITE(P_vmc_O, 8, 8, len - 1, temp + 1);
    SB_WAIT_ALL();

    //for (int j = 1; j < len; ++j) std::cout << temp[j] << " "; std::cout << "\n";

    //for (int j = 1; j < len; ++j)
    //  temp[j] = 0;
    //for (int k = 0; k < len; ++k) {
    //  for (int j = 1; j < len; ++j) {
    //    temp[j] += complex<float>(complex_conj_mul(hv[k], (i ? r : a)[k * len + j]));
    //  }
    //}

    SB_CONFIG(vv_config, vv_size);
    SB_DMA_READ((i ? r : a) + 1, 8 * len, 8 * (len - 1), len, P_vv_C);
    SB_DMA_READ(temp + 1, 0, 8 * (len - 1), len, P_vv_A);
    SB_DMA_WRITE(P_vv_O, 8, 8, (len - 1) * len, r);
    for (int j = 0; j < len; ++j) {
      SB_CONST(P_vv_B, *((uint64_t*)(hv + j)), len - 1);
    }
    SB_WAIT_ALL();
    
    //for (int j = 1; j < len; ++j)
    //  temp[j] = complex<float>(temp[j].real() * 2, temp[j].imag() * 2);
    //for (int j = 0; j < len; ++j) {
    //  for (int k = 1; k < len; ++k) {
    //    complex<float> delta(complex_mul(temp[k], hv[j]));
    //    r[j * (len - 1) + (k - 1)] = complex<float>(complex_sub((i ? r : a)[j * len + k], delta));
    //  }
    //}
    //for (int j = 0; j < len; ++j) { for (int k = 0; k < len - 1; ++k) std::cout << r[j * (len - 1) + k] << " "; std::cout << "\n"; }

    if (i != _N_ - 2) {
      --len;
      for (int j = 0; j < len; ++j)
        hv[j] = r[j];
      household(hv, len, f[i]);

      SB_CONFIG(mvc_config, mvc_size);
      SB_DMA_READ(hv, 0, 8 * len, len, P_mvc_A);
      SB_DMA_READ(r + len, 8, 8, len * len, P_mvc_B);
      for (int j = 0; j < len; ++j) {
        SB_CONST(P_mvc_reset, 0, len - 1);
        SB_CONST(P_mvc_reset, 1, 1);
        SB_GARBAGE(P_mvc_O, len - 1);
        SB_DMA_WRITE(P_mvc_O, 8, 8, 1, temp + j);
      }
      SB_WAIT_ALL();
      //for (int j = 0; j < len; ++j) {
      //  temp[j] = 0;
      //  for (int k = 0; k < len; ++k) {
      //    complex<float> delta(complex_conj_mul(hv[k], r[(j + 1) * len + k]));
      //    temp[j] = complex<float>(complex_add(temp[j], delta));
      //  }
      //}

      SB_CONFIG(vv_config, vv_size);
      SB_DMA_READ(r + len, 8, 8, len * len, P_vv_C);
      SB_DMA_READ(hv, 0, 8 * len, len, P_vv_A);
      SB_DMA_WRITE(P_vv_O, 8, 8, len * len, r);
      for (int j = 0; j < len; ++j) {
        SB_CONST(P_vv_B, *((uint64_t*)(temp + j)), len);
      }
      SB_WAIT_ALL();
      //for (int j = 0; j < len; ++j) {
      //  for (int k = 0; k < len; ++k) {
      //    complex<float> delta(complex_mul(temp[j], hv[k]));
      //    delta = complex<float>(complex_mul_cons(delta, 2));
      //    r[j * len + k] = complex<float>(complex_sub(r[(j + 1) * len + k], delta));
      //  }
      //}
      
      if (!i) {
        v[0] = complex<float>(1, 0);
        SB_CONFIG(vvc_config, vvc_size);
        SB_DMA_READ(hv, 0, 8 * (_N_ - 1), _N_ - 1, P_vvc_B);
        SB_DMA_WRITE(P_vvc_O, 8 * _N_, 8 * (_N_ - 1), _N_ - 1, v + _N_ + 1);
        for (int j = 1; j < _N_; ++j) {
          v[j] = v[j * _N_] = complex<float>(0, 0);
          SB_CONST(P_vvc_A, *((uint64_t*)(hv + j - 1)), _N_ - 1);
          SB_CONST(P_vvc_C, *((uint64_t*)&_one), 1);
          if (j != _N_ - 1) {
            SB_CONST(P_vvc_C, *((uint64_t*)&_zero), _N_ - 1);
          }
        }
        SB_WAIT_ALL();
        //for (int j = 1; j < _N_; ++j) {
        //  v[j] = v[j * _N_] = complex<float>(0, 0);
        //  for (int k = 1; k < _N_; ++k) {
        //    complex<float> delta(complex_conj_mul(hv[j - 1], hv[k - 1]));
        //    complex<float> diag(j == k, 0);
        //    complex<float> val(delta.real() * 2, delta.imag() * 2);
        //    v[j * _N_ + k] = complex<float>(complex_sub(diag, val));
        //  }
        //}
      } else {
        SB_CONFIG(vm_config, vm_size);
        SB_CONST(P_vm_C, *((uint64_t*)&_zero), _N_ - 1);
        SB_DMA_READ(v + _N_ * (i + 1) + 1, 8 * _N_, 8 * (_N_ - 1), len, P_vm_B);
        SB_RECURRENCE(P_vm_O, P_vm_C, (_N_ - 1) * (len - 1));
        SB_DMA_WRITE(P_vm_O, 8, 8, _N_ - 1, temp + 1);
        for (int j = i + 1; j < _N_; ++j) {
          SB_CONST(P_vm_A, hv + j - i - 1, _N_ - 1);
        }
        SB_WAIT_ALL();
        //for (int k = 1; k < _N_; ++k)
        //  temp[k] = 0;
        //for (int j = i + 1; j < _N_; ++j) {
        //  for (int k = 1; k < _N_; ++k) {
        //    complex<float> delta(complex_mul(hv[j - i - 1], v[j * _N_ + k]));
        //    temp[k] = complex<float>(complex_add(temp[k], delta));
        //  }
        //}
        SB_CONFIG(vvc_config, vvc_size);
        SB_DMA_READ(temp + 1, 0, 8 * (_N_ - 1), len, P_vvc_B);
        SB_DMA_READ(v + _N_ * (i + 1) + 1, 8 * _N_, 8 * (_N_ - 1), len, P_vvc_C);
        SB_DMA_WRITE(P_vvc_O, 8 * _N_, 8 * (_N_ - 1), len, v + _N_ * (i + 1) + 1);
        for (int k = 0; k < len; ++k) {
          SB_CONST(P_vvc_A, *((uint64_t*)(hv + k)), _N_ - 1);
        }
        SB_WAIT_ALL();
        //for (int k = 1; k < _N_; ++k)
        //  temp[k] = complex<float>(temp[k].real() * 2, temp[k].imag() * 2);
        //for (int k = 1; k < _N_; ++k) {
        //  for (int j = i + 1; j < _N_; ++j) {
        //    complex<float> delta(complex_conj_mul(hv[j - i - 1], temp[k]));
        //    v[j * _N_ + k] -= delta;
        //  }
        //}
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

  SB_CONFIG(mvc_config, mvc_size);
  SB_DMA_READ(v, 0, 8 * _N_ * _N_, _N_, P_mvc_A);
  for (int i = 0; i < _N_; ++i) {
    SB_DMA_READ(v + i * _N_, 0, 8 * _N_, _N_, P_mvc_B);
    for (int j = 0; j < _N_; ++j) {
      SB_CONST(P_mvc_reset, 0, _N_ - 1)
      SB_CONST(P_mvc_reset, 1, 1)
      SB_GARBAGE(P_mvc_O, _N_ - 1);
      SB_DMA_WRITE(P_mvc_O, 0, 8, 1, u + i * _N_ + j);
    }
  }
  SB_WAIT_ALL();

  //for (int i = 0; i < _N_; ++i) {
  //  for (int j = 0; j < _N_; ++j) {
  //    complex<float> sum(0, 0);
  //    for (int k = 0; k < _N_; ++k)
  //      sum += complex<float>(complex_conj_mul(v[j * _N_ + k], a[i * _N_ + k]));
  //    u[i * _N_ + j] = sum;
  //  }
  //}
  for (int i = 0; i < _N_; ++i) {
    for (int j = 0; j < _N_; ++j) {
      u[i * _N_ + j] /= s[j];
    }
  }
}
