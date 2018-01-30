#include "svd.h"
#include "sb_insts.h"
#include "matvec.h"
#include "vmc.h"
#include "vv.h"
#include "mvc.h"
#include "vvc.h"
#include "vm.h"
#include "lmm2x2.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_mul_cons(a, b) (a).real() * (b), (a).imag() * (b)

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

complex<float> f[N], d[N], r[N * N], temp[N];
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
  SB_CONST(P_lmm2x2_M0, ri_m0.val, N);
  SB_CONST(P_lmm2x2_M1, *((uint64_t*)&m1), N);
  SB_DMA_READ(v, 8, 8, N, P_lmm2x2_A);
  SB_DMA_READ(v + N, 8, 8, N, P_lmm2x2_B);
  SB_DMA_WRITE(P_lmm2x2_O0, 8, 8, N, v);
  SB_DMA_WRITE(P_lmm2x2_O1, 8, 8, N, v + N);
  SB_WAIT_ALL();
  //for (int i = 0; i < N; ++i) {
  //  lmm2x2(m0, m1, v[i], v[i + N]);
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
    SB_CONST(P_lmm2x2_M0, ri_m0.val, N);
    SB_CONST(P_lmm2x2_M1, *((uint64_t*)&m1), N);
    SB_DMA_READ(v + i * N, 8, 8, N, P_lmm2x2_A);
    SB_DMA_READ(v + i * N + N, 8, 8, N, P_lmm2x2_B);
    SB_DMA_WRITE(P_lmm2x2_O0, 8, 8, N, v + i * N);
    SB_DMA_WRITE(P_lmm2x2_O1, 8, 8, N, v + i * N + N);
    //for (int j = 0; j < N; ++j) {
    //  lmm2x2(m0, m1, v[i * N + j], v[(i + 1) * N + j]);
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
  for (int i = 0; i < N - 1; ++i) {
    int len = N - i;
    complex<float> hv[len], alpha;
    for (int j = 0; j < len; ++j)
      hv[j] = (i ? r : a)[j * len];
    household(hv, len, d[i]);

    REVELvec_mul_mat(0, len, 1, len, len, hv, true, (i ? r : a), temp + 1);

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
    SB_REPEAT_PORT(len - 1);
    SB_DMA_READ(hv, 8, 8, len, P_vv_B);
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

    if (i != N - 2) {
      --len;
      for (int j = 0; j < len; ++j)
        hv[j] = r[j];
      household(hv, len, f[i]);
      
      REVELmat_mul_vec(1, len, 0, len, len, r, hv, true, temp);
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
      SB_REPEAT_PORT(len);
      SB_DMA_READ(temp, 8, 8, len, P_vv_B);
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
        SB_DMA_READ(hv, 0, 8 * (N - 1), N - 1, P_vvc_B);
        SB_DMA_WRITE(P_vvc_O, 8 * N, 8 * (N - 1), N - 1, v + N + 1);
        SB_REPEAT_PORT(N - 1);
        SB_DMA_READ(hv, 8, 8, N - 1, P_vvc_A);
        SB_2D_CONST(P_vvc_C, *((uint64_t*)&_one), 1, *((uint64_t*)&_zero), N - 1, N - 2);
        SB_CONST(P_vvc_C, *((uint64_t*)&_one), 1);
        for (int j = 1; j < N; ++j) {
          v[j] = v[j * N] = complex<float>(0, 0);
        }
        SB_WAIT_ALL();
        //for (int j = 1; j < N; ++j) {
        //  v[j] = v[j * N] = complex<float>(0, 0);
        //  for (int k = 1; k < N; ++k) {
        //    complex<float> delta(complex_conj_mul(hv[j - 1], hv[k - 1]));
        //    complex<float> diag(j == k, 0);
        //    complex<float> val(delta.real() * 2, delta.imag() * 2);
        //    v[j * N + k] = complex<float>(complex_sub(diag, val));
        //  }
        //}
      } else {
        SB_CONFIG(vm_config, vm_size);
        SB_CONST(P_vm_C, *((uint64_t*)&_zero), N - 1);
        SB_DMA_READ(v + N * (i + 1) + 1, 8 * N, 8 * (N - 1), len, P_vm_B);
        SB_RECURRENCE(P_vm_O, P_vm_C, (N - 1) * (len - 1));
        SB_DMA_WRITE(P_vm_O, 8, 8, N - 1, temp + 1);
        for (int j = i + 1; j < N; ++j) {
          SB_CONST(P_vm_A, hv + j - i - 1, N - 1);
        }
        SB_WAIT_ALL();
        //for (int k = 1; k < N; ++k)
        //  temp[k] = 0;
        //for (int j = i + 1; j < N; ++j) {
        //  for (int k = 1; k < N; ++k) {
        //    complex<float> delta(complex_mul(hv[j - i - 1], v[j * N + k]));
        //    temp[k] = complex<float>(complex_add(temp[k], delta));
        //  }
        //}
        SB_CONFIG(vvc_config, vvc_size);
        SB_DMA_READ(temp, 0, 8 * (N - 1), len, P_vvc_B);
        SB_DMA_READ(v + N * (i + 1) + 1, 8 * N, 8 * (N - 1), len, P_vvc_C);
        SB_DMA_WRITE(P_vvc_O, 8 * N, 8 * (N - 1), len, v + N * (i + 1) + 1);
        for (int k = 0; k < len; ++k) {
          SB_CONST(P_vvc_A, *((uint64_t*)(hv + k)), N - 1);
        }
        SB_WAIT_ALL();
        //for (int k = 1; k < N; ++k)
        //  temp[k] = complex<float>(temp[k].real() * 2, temp[k].imag() * 2);
        //for (int k = 1; k < N; ++k) {
        //  for (int j = i + 1; j < N; ++j) {
        //    complex<float> delta(complex_conj_mul(hv[j - i - 1], temp[k]));
        //    v[j * N + k] -= delta;
        //  }
        //}
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

  SB_CONFIG(mvc_config, mvc_size);
  SB_DMA_READ(v, 0, 8 * N * N, N, P_mvc_A);
  for (int i = 0; i < N; ++i) {
    SB_DMA_READ(v + i * N, 0, 8 * N, N, P_mvc_B);
    for (int j = 0; j < N; ++j) {
      SB_CONST(P_mvc_reset, 0, N - 1)
      SB_CONST(P_mvc_reset, 1, 1)
      SB_GARBAGE(P_mvc_O, N - 1);
      SB_DMA_WRITE(P_mvc_O, 0, 8, 1, u + i * N + j);
    }
  }
  SB_WAIT_ALL();

  //for (int i = 0; i < N; ++i) {
  //  for (int j = 0; j < N; ++j) {
  //    complex<float> sum(0, 0);
  //    for (int k = 0; k < N; ++k)
  //      sum += complex<float>(complex_conj_mul(v[j * N + k], a[i * N + k]));
  //    u[i * N + j] = sum;
  //  }
  //}
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      u[i * N + j] /= s[j];
    }
  }
}
