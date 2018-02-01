#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "matvec.h"
#include "sub2outer.h"
#include "sub2couter.h"
#include "multi.h"

static complex_t _one = {1, 0}, _zero = {0, 0};

static complex_t temp[N], hv[N];

void qr(complex<float> *a, complex<float> *q, complex<float> *r) {
  for (int i = 0; i < N; ++i) {
    int n  = N - i;
    complex_t alpha;
    complex<float> *m = (i ? r : a);
    if (n != 1) {
      SB_CONFIG(multi_config, multi_size);
      int pad = get_pad(n - 1, 4);
      union {
        float a[2];
        uint64_t b;
      } _norm1, to_sqrt, head;
      head.a[0] = m[i * N + i].real();
      head.a[1] = m[i * N + i].imag();
      SB_FILL_MODE(POST_ZERO_FILL);
      SB_DMA_READ(m + (i + 1) * N + i, 8 * N, 8, n - 1, P_multi_V);
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
      SB_DMA_READ(m + (i + 1) * N + i, 8 * N, 8, n - 1, P_multi_VEC);
      pad = get_pad(n, 4);
      SB_CONST(P_multi_norm, to_sqrt.b, (n + pad) / 4);
      SB_DMA_WRITE(P_multi_O, 8, 8, n, hv);
      SB_GARBAGE(P_multi_O, pad);
      SB_WAIT_ALL();
    } else {
      float norm = sqrt(complex_norm(m[i * N + i]));
      hv[0] = (complex_t) {m[i * N + i].real() / norm, m[i * N + i].imag() / norm};
      alpha = (complex_t) {-m[i * N + i].real(), -m[i * N + i].imag()};
    }
    
    REVELvec_mul_mat(i, n, i, n, N, hv, true, (i ? r : a), temp);

    int pad = get_pad(n - 1, 4);
    SB_CONFIG(sub2outer_config, sub2outer_size);
    SB_FILL_MODE(STRIDE_ZERO_FILL);
    SB_DMA_READ((i ? r : a) + i * N + i + 1, 8 * N, 8 * (n - 1), n, P_sub2outer_A);
    SB_REPEAT_PORT((n - 1 + pad) / 4);
    SB_DMA_READ(hv, 8, 8, n, P_sub2outer_B);
    SB_DMA_READ(temp + 1, 0, 8 * (n - 1), n, P_sub2outer_C);
    for (int j = i; j < N; ++j) {
      r[j * N + i] = (j == i) ? complex<float>(alpha.real, alpha.imag) : 0;
      SB_DMA_WRITE(P_sub2outer_O, 8, 8, n - 1, r + j * N + i + 1);
      SB_GARBAGE(P_sub2outer_O, pad);
      //for (int k = i + 1; k < N; ++k)
      //  r[j * N + k] = (i ? r : a)[j * N + k] - hv[j - i] * temp[k - i] * 2.0f;
    }
    SB_FILL_MODE(NO_FILL);
    SB_WAIT_ALL();

    if (i) {
      REVELmat_mul_vec(0, N, i, n, N, q, hv, false, temp);

      int pad = get_pad(n, 4);
      SB_CONFIG(sub2couter_config, sub2couter_size);
      SB_FILL_MODE(STRIDE_ZERO_FILL);
      SB_DMA_READ(q + i, 8 * N, 8 * n, N, P_sub2couter_A);
      SB_REPEAT_PORT((n + pad) / 4);
      SB_DMA_READ(temp, 8, 8, N, P_sub2couter_B);
      SB_DMA_READ(hv, 0, 8 * n, N, P_sub2couter_C);
      for (int j = 0; j < N; ++j) {
        SB_DMA_WRITE(P_sub2couter_O, 8, 8, n, q + j * N + i);
        SB_GARBAGE(P_sub2couter_O, pad);
        //for (int k = i; k < N; ++k)
        //  q[j * N + k] -= temp[j] * std::conj(hv[k - i]) * 2.0f;
      }
      SB_WAIT_ALL();

    } else {

      SB_CONFIG(sub2couter_config, sub2couter_size);
      SB_FILL_MODE(NO_FILL);
      SB_2D_CONST(P_sub2couter_A, *((uint64_t*)&_one), 1, *((uint64_t*) &_zero), N, N - 1);
      SB_CONST(P_sub2couter_A, *((uint64_t*) &_one), 1);
      SB_REPEAT_PORT(N / 4);
      SB_DMA_READ(hv, 8, 8, N, P_sub2couter_B);
      SB_DMA_READ(hv, 0, 8 * N, N, P_sub2couter_C);
      SB_DMA_WRITE(P_sub2couter_O, 8 * N, 8 * N, N, q);
      SB_WAIT_ALL();
      //for (int j = 0; j < N; ++j)
        //for (int k = 0; k < N; ++k)
          //q[j * N + k] = (j == k ? 1.0f : 0.0f) - std::conj(hv[k]) * hv[j] * 2.0f;
    }
  }
}
#undef h

