#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "matvec.h"
#include "multi.dfg.h"

static complex_t _one = {1, 0}, _zero = {0, 0};

static complex_t temp[N], hv[N];
static complex_t alpha;

void qr(complex<float> *a, complex<float> *q, complex<float> *r) {
  {
    int i = 0;
    int n  = N - i;
    complex<float> *m = a;
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
 
    REVELvec_mul_mat(m + i * N + i, n, n, N, hv, true, temp);
    REVELsub_outerx2(m + i * N + i, n, n, N, hv, temp, false, r + i * N + i, N);
    //for (int j = i; j < N; ++j) {
    //  r[j * N + i] = (j == i) ? complex<float>(alpha.real, alpha.imag) : 0;
    //  for (int k = i + 1; k < N; ++k)
    //    r[j * N + k] = (i ? r : a)[j * N + k] - hv[j - i] * temp[k - i] * 2.0f;
    //}
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
 
    REVELvec_mul_mat(m + i * N + i, n, n, N, hv, true, temp);
    REVELsub_outerx2(m + i * N + i, n, n, N, hv, temp, false, r + i * N + i, N);
    //for (int j = i; j < N; ++j) {
    //  r[j * N + i] = (j == i) ? complex<float>(alpha.real, alpha.imag) : 0;
    //  for (int k = i + 1; k < N; ++k)
    //    r[j * N + k] = (i ? r : a)[j * N + k] - hv[j - i] * temp[k - i] * 2.0f;
    //}

    REVELmat_mul_vec(q + i, N, n, N, hv, false, temp);
    REVELsub_outerx2(q + i, N, n, N, temp, hv, true, q + i, N);
    //for (int j = 0; j < N; ++j)
    //  for (int k = i; k < N; ++k)
    //    q[j * N + k] -= temp[j] * std::conj(hv[k - i]) * 2.0f;
  }
}
#undef h

