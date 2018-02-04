#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "mulconj.dfg.h"
#include "finalize_compact.dfg.h"
#include "norm.dfg.h"

#define complex_mul(a, b) (complex_t) { \
  (a).real * (b).real - (a).imag * (b).imag, \
  (a).real * (b).imag + (a).imag * (b).real }

#define complex_conj_mul(a, b) (complex_t) { \
  (a).real * (b).real + (a).imag * (b).imag, \
  (a).real * (b).imag - (a).imag * (b).real }

#define complex_add(a, b) (complex_t) { (a).real + (b).real, (a).imag + (b).imag }

#define complex_sub(a, b) (complex_t) { (a).real - (b).real, (a).imag - (b).imag }

#define complex_norm(a) ((a).real * (a).real + (a).imag * (a).imag)

complex_t tmp0[N * N];
complex_t tmp1[N * N];
complex_t sub_q[N * N];
//double buffering
complex_t rbuffer0[N * N];

union _reinterpret_t {
  complex_t a;
  uint64_t val;
};

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int i, j, k, x, y;

  complex_t *r = rbuffer0;

  for (i = 0; i < N; ++i) {
    int len = N - i;
// Household Vector
    complex_t v[len];
    {
      if (len > 1) {
        float norm1;
        SB_CONFIG(norm_config, norm_size);
        if (i) {
          SB_DMA_READ(r + 1, 8, 8, len - 1, P_norm_A);
          SB_DMA_READ(r + 1, 8, 8, len - 1, P_norm_B);
        } else {
          SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_norm_A);
          SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_norm_B);
        }
        SB_CONST(P_norm_C, 0, len - 1);
        SB_CONST(P_norm_reset, 2, len - 2);
        SB_CONST(P_norm_reset, 1, 1);
        SB_CONST(P_norm_sqrt, 0, len - 1);

        complex_t head = i ? *r : (complex_t) {a->real(), a->imag()};
        float norm0 = 1. / (head.real * head.real + head.imag * head.imag);

        SB_RECV(P_norm_O0, norm1);

        _reinterpret_t val = {1 + norm1 * norm0, 0}, another;
        SB_CONST(P_norm_A, 0, 1);
        SB_CONST(P_norm_B, 0, 1);
        SB_CONST(P_norm_reset, 3, 1);
        SB_CONST(P_norm_sqrt, 1, 1);
        SB_CONST(P_norm_C, val.val, 1);

        float rate;
        SB_RECV(P_norm_O1, rate);
        head.real *= rate + 1;
        head.imag *= rate + 1;

        val.a.real = head.real * head.real + head.imag * head.imag + norm1;
        SB_CONST(P_norm_A, 0, 1);
        SB_CONST(P_norm_B, 0, 1);
        SB_CONST(P_norm_reset, 3, 1);
        SB_CONST(P_norm_sqrt, 1, 1);
        SB_CONST(P_norm_C, val.val, 1);
        SB_RECV(P_norm_O1, norm0);


        val.a.real = 1 / norm0;
        SB_CONST(P_norm_B, *((uint64_t*)&head), 1);
        if (i) {
          SB_DMA_READ(r + 1, 8, 8, len - 1, P_norm_B);
        } else {
          SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_norm_B);
        }
        SB_CONST(P_norm_A, val.val, len);
        SB_CONST(P_norm_C, 0, len);
        SB_CONST(P_norm_reset, 1, len);
        SB_CONST(P_norm_sqrt, 0, len);
        SB_DMA_WRITE(P_norm_O0, 8, 8, len, v);

        SB_WAIT_ALL();
      } else {
        if (N == 1) {
          *v = (complex_t) {a->real(), a->imag()};
        } else {
          *v = *r;
        }
        float norm = 1 / sqrt(complex_norm(*v));
        v->real *= norm;
        v->imag *= norm;
      }
    }

// Intermediate result computing
    {
      complex_t *tmpxq = tmp0;
      complex_t *tmpxr = tmp1;

      SB_CONFIG(mulconj_config, mulconj_size);
      if (i) {
        SB_DMA_READ(sub_q, 0, N * len * 8, 1, P_mulconj_Q);
      } else {
        _reinterpret_t _one = {1, 0}, _zero = {0, 0};
        SB_2D_CONST(P_mulconj_Q, _one.val, 1, _zero, N, N - 1);
        SB_CONST(P_mulconj_Q, _one.val, 1);
      }
      SB_DMA_READ(v, 0, 8 * len, N, P_mulconj_V);
      SB_CONST(P_mulconj_R, 0, i * len);

      SB_GARBAGE(P_mulconj_O1, i);
      SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, N);
      SB_DMA_WRITE(P_mulconj_O0, 8, 8, N, tmpxq);
      SB_DMA_WRITE(P_mulconj_O1, 8, 8, len, tmpxr);

      if (i) {
        SB_DMA_READ(r, 0, len * len * 8, 1, P_mulconj_R);
      } else {
        for (int j = 0; j < N; ++j) {
          SB_DMA_READ(a + j, 8 * N, 8, N, P_mulconj_R);
        }
      }


      SB_WAIT_ALL();
    }

// Finalize the result
    {
      complex_t *tmpxq = tmp0;

      SB_CONFIG(finalize_compact_config, finalize_compact_size);
      //SB_CONST(P_finalize_compact_W, *((uint64_t*)&w), N * len);
      SB_DMA_READ(v, 0, 8 * len, N, P_finalize_compact_VQ); 
      SB_CONST(P_finalize_compact_TMPR, 0, i * len);
      SB_CONST(P_finalize_compact_VR, 0, i * len);
      SB_CONST(P_finalize_compact_R, 0, i * len);
      if (i) {
        SB_DMA_READ(sub_q, 0, N * len * 8, 1, P_finalize_compact_Q);
      } else {
        _reinterpret_t _one = {1, 0}, _zero = {0, 0};
        SB_2D_CONST(P_finalize_compact_Q, _one.val, 1, _zero, N, N - 1);
        SB_CONST(P_finalize_compact_Q, _one.val, 1);
      }
      SB_GARBAGE(P_finalize_compact_O1, i * len);

      SB_REPEAT_PORT(len);
      SB_DMA_READ(tmpxq, 8, 8, N, P_finalize_compact_TMPQ);

      SB_2D_CONST(P_finalize_compact_MUX0, 0, 1, 1, len - 1, N);
      SB_DMA_WRITE(P_finalize_compact_O00, 8 * N, 8, N, Q + i);
      SB_DMA_WRITE(P_finalize_compact_O01, 8, 8, (len - 1) * N, sub_q);

      {
        SB_DMA_READ(tmp1, 0, 8 * len, 1, P_finalize_compact_TMPR);
        SB_CONST(P_finalize_compact_VR, *((uint64_t*)v), len);
        if (i) {
          SB_DMA_READ(r, len * 8, 8, len, P_finalize_compact_R);
        } else {
          SB_DMA_READ(a, 8, 8, N, P_finalize_compact_R);
        }

        SB_DMA_WRITE(P_finalize_compact_O1, 8, 8, len, R + i * N + i);
      }


      SB_CONST(P_finalize_compact_TMPR, 0, len - 1);
      SB_CONST(P_finalize_compact_VR, 0, len - 1);
      SB_CONST(P_finalize_compact_R, 0, len - 1);
      SB_GARBAGE(P_finalize_compact_O1, len - 1);

      SB_DMA_READ(v + 1, 0, 8 * (len - 1), len - 1, P_finalize_compact_VR);
      SB_DMA_WRITE(P_finalize_compact_O1, 8, 8, (len - 1) * (len - 1), r);


      SB_REPEAT_PORT(len - 1);
      SB_DMA_READ(tmp1 + 1, 8, 8, len - 1, P_finalize_compact_TMPR);

      if (i) {
        SB_DMA_READ(r + len + 1, 8 * len, 8 * (len - 1), len - 1, P_finalize_compact_R);
      } else {
        for (int j = 1; j < N; ++j) {
          SB_DMA_READ(a + N + j, 8 * N, 8, N - 1, P_finalize_compact_R);
        }
      }

      SB_WAIT_ALL();
    }

  }
}
#undef h

