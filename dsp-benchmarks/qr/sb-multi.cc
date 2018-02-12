#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "matvec.h"
#include "sub2outer.h"
#include "sub2couter.h"
#include "multi.dfg.h"

static complex_t _one = {1, 0}, _zero = {0, 0};

static complex_t temp[_N_], hv[_N_];

void qr(complex<float> *a, complex<float> *q, complex<float> *r) {
  for (int i = 0; i < _N_; ++i) {
    int n  = _N_ - i;
    complex_t alpha;
    complex<float> *m = (i ? r : a);
    if (n != 1) {
      SB_CONFIG(multi_config, multi_size);
      int pad = get_pad(n - 1, 4);
      union {
        float a[2];
        uint64_t b;
      } _norm1, to_sqrt, head;
      head.a[0] = m[i * _N_ + i].real();
      head.a[1] = m[i * _N_ + i].imag();
      SB_FILL_MODE(POST_ZERO_FILL);
      SB_DMA_READ(m + (i + 1) * _N_ + i, 8 * _N_, 8, n - 1, P_multi_V);
      SB_CONST(P_multi_reset, 2, (n + pad - 1) / 4 - 1);
      SB_CONST(P_multi_reset, 1, 1);
      float norm0 = head.a[0] * head.a[0] + head.a[1] * head.a[1];
      SB_RECV(P_multi_NORM2, _norm1.b);

      float norm1 = _norm1.a[0] + _norm1.a[1];
      to_sqrt.a[0] = 1 + norm1 / norm0;
      SB_CONST(P_multi_sqrt, to_sqrt.b, 1);
      SB_RECV(P_multi_NORM, to_sqrt.b);

      alpha = (complex_t) {-head.a[0] * to_sqrt.a[0], -head.a[1] * to_sqrt.a[0]};
      float rate = 1 + to_sqrt.a[0];
      head.a[0] *= rate;
      head.a[1] *= rate;
      norm1 += head.a[0] * head.a[0] + head.a[1] * head.a[1];

      to_sqrt.a[0] = norm1;
      SB_CONST(P_multi_sqrt, to_sqrt.b, 1);
      SB_RECV(P_multi_NORM, to_sqrt.b);
      to_sqrt.a[0] = to_sqrt.a[1] = 1 / to_sqrt.a[0];

      SB_FILL_MODE(NO_FILL);
      SB_CONST(P_multi_VEC, head.b, 1);
      SB_FILL_MODE(POST_ZERO_FILL);
      SB_DMA_READ(m + (i + 1) * _N_ + i, 8 * _N_, 8, n - 1, P_multi_VEC);
      pad = get_pad(n, 4);
      SB_CONST(P_multi_norm, to_sqrt.b, (n + pad) / 4);
      SB_DMA_WRITE(P_multi_O, 8, 8, n, hv);
      SB_GARBAGE(P_multi_O, pad);
      SB_WAIT_ALL();
    } else {
      float norm = sqrt(complex_norm(m[i * _N_ + i]));
      hv[0] = (complex_t) {m[i * _N_ + i].real() / norm, m[i * _N_ + i].imag() / norm};
      alpha = (complex_t) {-m[i * _N_ + i].real(), -m[i * _N_ + i].imag()};
    }
    
    REVELvec_mul_mat(i, n, i, n, _N_, hv, true, (i ? r : a), temp);

    int pad = get_pad(n - 1, 4);
    SB_CONFIG(sub2outer_config, sub2outer_size);
    SB_FILL_MODE(STRIDE_ZERO_FILL);
    SB_DMA_READ((i ? r : a) + i * _N_ + i + 1, 8 * _N_, 8 * (n - 1), n, P_sub2outer_A);
    SB_REPEAT_PORT((n - 1 + pad) / 4);
    SB_DMA_READ(hv, 8, 8, n, P_sub2outer_B);
    SB_DMA_READ(temp + 1, 0, 8 * (n - 1), n, P_sub2outer_C);
    for (int j = i; j < _N_; ++j) {
      r[j * _N_ + i] = (j == i) ? complex<float>(alpha.real, alpha.imag) : 0;
      SB_DMA_WRITE(P_sub2outer_O, 8, 8, n - 1, r + j * _N_ + i + 1);
      SB_GARBAGE(P_sub2outer_O, pad);
      //for (int k = i + 1; k < _N_; ++k)
      //  r[j * _N_ + k] = (i ? r : a)[j * _N_ + k] - hv[j - i] * temp[k - i] * 2.0f;
    }
    SB_FILL_MODE(NO_FILL);
    SB_WAIT_ALL();

    if (i) {
      REVELmat_mul_vec(0, _N_, i, n, _N_, q, hv, false, temp);

      int pad = get_pad(n, 4);
      SB_CONFIG(sub2couter_config, sub2couter_size);
      SB_FILL_MODE(STRIDE_ZERO_FILL);
      SB_DMA_READ(q + i, 8 * _N_, 8 * n, _N_, P_sub2couter_A);
      SB_REPEAT_PORT((n + pad) / 4);
      SB_DMA_READ(temp, 8, 8, _N_, P_sub2couter_B);
      SB_DMA_READ(hv, 0, 8 * n, _N_, P_sub2couter_C);
      for (int j = 0; j < _N_; ++j) {
        SB_DMA_WRITE(P_sub2couter_O, 8, 8, n, q + j * _N_ + i);
        SB_GARBAGE(P_sub2couter_O, pad);
        //for (int k = i; k < _N_; ++k)
        //  q[j * _N_ + k] -= temp[j] * std::conj(hv[k - i]) * 2.0f;
      }
      SB_WAIT_ALL();

    } else {

      SB_CONFIG(sub2couter_config, sub2couter_size);
      SB_FILL_MODE(NO_FILL);
      SB_2D_CONST(P_sub2couter_A, *((uint64_t*)&_one), 1, *((uint64_t*) &_zero), _N_, _N_ - 1);
      SB_CONST(P_sub2couter_A, *((uint64_t*) &_one), 1);
      SB_REPEAT_PORT(_N_ / 4);
      SB_DMA_READ(hv, 8, 8, _N_, P_sub2couter_B);
      SB_DMA_READ(hv, 0, 8 * _N_, _N_, P_sub2couter_C);
      SB_DMA_WRITE(P_sub2couter_O, 8 * _N_, 8 * _N_, _N_, q);
      SB_WAIT_ALL();
      //for (int j = 0; j < _N_; ++j)
        //for (int k = 0; k < _N_; ++k)
          //q[j * _N_ + k] = (j == k ? 1.0f : 0.0f) - std::conj(hv[k]) * hv[j] * 2.0f;
    }
  }
}
#undef h

