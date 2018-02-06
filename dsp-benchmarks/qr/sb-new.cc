#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "matvec.h"
#include "multi.dfg.h"
#include "fused1.dfg.h"
#include "fused2.dfg.h"

static complex_t _one = {1, 0}, _zero = {0, 0};

static complex_t temp0[N], temp1[N];
static complex_t hv[N];
static complex_t alpha;

void qr(complex<float> *a, complex<float> *q, complex<float> *r) {
  {
    SB_CONFIG(multi_config, multi_size);
    int pad = get_pad(N - 1, 4);
    SB_FILL_MODE(POST_ZERO_FILL);
    SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_multi_V);
    SB_CONST(P_multi_reset, 2, (N + pad - 1) / 4 - 1);
    SB_CONST(P_multi_reset, 1, 1);
    union {
      float a[2];
      uint64_t b;
    } _norm1, to_sqrt, head;
    head.a[0] = a->real();
    head.a[1] = a->imag();
    float norm0 = head.a[0] * head.a[0] + head.a[1] * head.a[1];
    SB_RECV(P_multi_NORM2, _norm1.b);

    float norm1 = _norm1.a[0] + _norm1.a[1];
    float _alpha = sqrt(1 + norm1 / norm0);

    alpha = (complex_t) {-head.a[0] * _alpha, -head.a[1] * _alpha};
    float rate = 1 + _alpha;
    head.a[0] *= rate;
    head.a[1] *= rate;
    norm1 += head.a[0] * head.a[0] + head.a[1] * head.a[1];

    to_sqrt.a[0] = to_sqrt.a[1] = 1 / sqrt(norm1);

    SB_FILL_MODE(NO_FILL);
    SB_CONST(P_multi_VEC, head.b, 1);
    SB_FILL_MODE(POST_ZERO_FILL);
    SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_multi_VEC);
    pad = get_pad(N, 4);
    SB_CONST(P_multi_norm, to_sqrt.b, (N + pad) / 4);
    SB_DMA_WRITE(P_multi_O, 8, 8, N, hv);
    SB_GARBAGE(P_multi_O, pad);
    SB_WAIT_ALL();
 
    REVELvec_mul_mat(a, N, N, N, hv, true, temp0);
    REVELsub_outerx2(a, N, N, N, hv, temp0, false, r, N);
    REVELsub_outerx2(NULL, N, N, -1, hv, hv, true, q, N);
  }
  for (int i = 1; i < N - 1; ++i) {
    int n  = N - i;
    complex<float> *m = r;

    SB_CONFIG(multi_config, multi_size);
    int pad = get_pad(n - 1, 4);
    SB_FILL_MODE(POST_ZERO_FILL);
    SB_DMA_READ(m + (i + 1) * N + i, 8 * N, 8, n - 1, P_multi_V);
    SB_CONST(P_multi_reset, 2, (n + pad - 1) / 4 - 1);
    SB_CONST(P_multi_reset, 1, 1);
    union {
      float a[2];
      uint64_t b;
    } _norm1, to_sqrt, head;
    head.a[0] = m[i * N + i].real();
    head.a[1] = m[i * N + i].imag();
    float norm0 = head.a[0] * head.a[0] + head.a[1] * head.a[1];
    SB_RECV(P_multi_NORM2, _norm1.b);

    float norm1 = _norm1.a[0] + _norm1.a[1];
    float _alpha = sqrt(1 + norm1 / norm0);

    alpha = (complex_t) {-head.a[0] * _alpha, -head.a[1] * _alpha};
    float rate = 1 + _alpha;
    head.a[0] *= rate;
    head.a[1] *= rate;
    norm1 += head.a[0] * head.a[0] + head.a[1] * head.a[1];

    to_sqrt.a[0] = to_sqrt.a[1] = 1 / sqrt(norm1);

    SB_FILL_MODE(NO_FILL);
    SB_CONST(P_multi_VEC, head.b, 1);
    SB_FILL_MODE(POST_ZERO_FILL);
    SB_DMA_READ(m + (i + 1) * N + i, 8 * N, 8, n - 1, P_multi_VEC);
    pad = get_pad(n, 4);
    SB_CONST(P_multi_norm, to_sqrt.b, (n + pad) / 4);
    SB_DMA_WRITE(P_multi_O, 8, 8, n, hv);
    SB_GARBAGE(P_multi_O, pad);
    SB_WAIT_ALL();
 ////////////////////////////////////////////////////////////////////////
    SB_CONFIG(fused1_config, fused1_size);

    //REVELvec_mul_mat(m + i * N + i, n, n, N, hv, true, temp0);
    {
      int pad = get_pad(n, 2);
      SB_FILL_MODE(STRIDE_ZERO_FILL);
      SB_DMA_READ(m + i * N + i, 8 * N, 8 * n, n, P_fused1_A);
      SB_CONST(P_fused1_C, 0, n + pad);
      SB_RECURRENCE(P_fused1_O, P_fused1_C, (n + pad) * (n - 1));
      SB_REPEAT_PORT((n + pad) / 2);
      SB_DMA_READ(hv, 8, 8, n, P_fused1_B);
      SB_DMA_WRITE(P_fused1_O, 8, 8, n, temp0);
      SB_GARBAGE(P_fused1_O, pad);
    }

    //REVELmat_mul_vec(q + i, N, n, N, hv, false, temp1);
    {
      int pad = get_pad(n, 2);
      SB_FILL_MODE(STRIDE_ZERO_FILL);
      SB_DMA_READ(q + i, 8 * N, 8 * n, N, P_fused1_A_);
      SB_DMA_READ(hv, 0, 8 * n, N, P_fused1_B_);
      SB_2D_CONST(P_fused1_reset, 2, (n + pad) / 2 - 1, 1, 1, N);
      SB_DMA_WRITE(P_fused1_O_, 8, 8, N, temp1);
    }
    SB_WAIT_ALL();

    //REVELsub_outerx2(m + i * N + i, n, n, N, hv, temp0, false, r + i * N + i, N);
    //for (int j = i; j < N; ++j) {
    //  r[j * N + i] = (j == i) ? complex<float>(alpha.real, alpha.imag) : 0;
    //  for (int k = i + 1; k < N; ++k)
    //    r[j * N + k] = (i ? r : a)[j * N + k] - hv[j - i] * temp[k - i] * 2.0f;
    //}
    SB_CONFIG(fused2_config, fused2_size);
    {
      SB_DMA_READ(r + i * N + i, 8 * N, 8 * n, n, P_fused2_A_);
      SB_REPEAT_PORT(n);
      SB_DMA_READ(hv, 8, 8, n, P_fused2_B_);
      SB_DMA_READ(temp0, 0, 8 * n, n, P_fused2_C_);
      SB_DMA_WRITE(P_fused2_O_, 8 * N, 8 * n, n, r + i * N + i);
    }

    //REVELsub_outerx2(q + i, N, n, N, temp1, hv, true, q + i, N);
    //for (int j = 0; j < N; ++j)
    //  for (int k = i; k < N; ++k)
    //    q[j * N + k] -= temp[j] * std::conj(hv[k - i]) * 2.0f;
    {
      int pad = get_pad(n, 2);
      SB_FILL_MODE(STRIDE_DISCARD_FILL);
      SB_DMA_READ(q + i, 8 * N, 8 * n, N, P_fused2_A);
      SB_REPEAT_PORT((n + pad) / 2);
      SB_DMA_READ(temp1, 8, 8, N, P_fused2_B);
      SB_DMA_READ(hv, 0, 8 * n, N, P_fused2_C);
      SB_DMA_WRITE(P_fused2_O, 8 * N, 8 * n, N, q + i);
      SB_WAIT_ALL();
    }
  }
}
#undef h

