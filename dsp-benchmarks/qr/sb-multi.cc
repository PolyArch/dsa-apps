#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "mulconj.h"
#include "finalize_compact.h"
#include "conjconj.h"

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

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int i, j, k, x, y;
  for (i = 0; i < N * N; ++i) {
    sub_q[i] = i % N == i / N ? (complex_t){1, 0} : (complex_t){0, 0};
  }

  complex_t *r = rbuffer0;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      r[j * N + i] = (complex_t){a[i * N + j].real(), a[i * N + j].imag()};
      //r[j * N + i] = a[i * N + j];
    }
  }

  SB_CONTEXT(1);
  SB_CONFIG(conjconj_config, conjconj_size);
  SB_CONTEXT(2);
  SB_CONFIG(mulconj_config, mulconj_size);
  SB_CONTEXT(4);
  SB_CONFIG(finalize_compact_config, finalize_compact_size);

  for (i = 0; i < N; ++i) {
    int len = N - i;
// Household Vector
    complex_t v[len];
    complex_t head;
    {
      float norm = 0;

      //SB_CONFIG(conjconj_config, conjconj_size);
      SB_CONTEXT(1);
      SB_DMA_READ(r, 8, 8, len, P_conjconj_A0);
      SB_DMA_READ(r, 8, 8, len, P_conjconj_B0);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_CONST(P_conjconj_sqrt, 0, len - 1);
      SB_CONST(P_conjconj_sqrt, 1, 1);

      SB_GARBAGE(P_conjconj_O0, 1);
      SB_GARBAGE(P_conjconj_O1, 1);
      SB_RECV(P_conjconj_O2, norm);

      complex_t temp = *r;
      float sign = sqrt(temp.real * temp.real + temp.imag * temp.imag);
      head.real = temp.real + temp.real / sign * norm;
      head.imag = temp.imag + temp.imag / sign * norm;
    }

    {
      float norm = 0;
      SB_CONST(P_conjconj_A0, *((uint64_t *) &head), 1);
      SB_CONST(P_conjconj_B0, *((uint64_t *) &head), 1);
      SB_DMA_READ(r + 1, 8, 8, len - 1, P_conjconj_A0);
      SB_DMA_READ(r + 1, 8, 8, len - 1, P_conjconj_B0);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_CONST(P_conjconj_sqrt, 0, len - 1);
      SB_CONST(P_conjconj_sqrt, 1, 1);

      SB_GARBAGE(P_conjconj_O0, 1);
      SB_GARBAGE(P_conjconj_O1, 1);
      SB_RECV(P_conjconj_O2, norm);

      union {
        complex_t a;
        uint64_t b;
      } ri = {(float)(1. / norm), 0};

      SB_CONST(P_conjconj_B0, *((uint64_t *) &head), 1);
      SB_DMA_READ(r + 1, 8, 8, len - 1, P_conjconj_B0);
      SB_CONST(P_conjconj_A0, ri.b, len);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      SB_CONST(P_conjconj_reset, 1, len);
      SB_CONST(P_conjconj_sqrt, 0, len);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, len, v);
      SB_GARBAGE(P_conjconj_O1, len);

      SB_WAIT_ALL();

    }

    complex_t w;
    {
      complex_t xv = {0, 0}, vx = {0, 0};
      complex_t *rp = r,  *vp = v;

      SB_DMA_READ(r, 8, 8, len, P_conjconj_A0);
      SB_DMA_READ(v, 8, 8, len, P_conjconj_B0);
      SB_DMA_READ(v, 8, 8, len, P_conjconj_A1);
      SB_DMA_READ(r, 8, 8, len, P_conjconj_B1);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_CONST(P_conjconj_sqrt, 0, len);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, 1, &xv);
      SB_DMA_WRITE(P_conjconj_O1, 8, 8, 1, &vx);
      SB_WAIT_ALL();

      float norm = 1 / complex_norm(vx);
      w = complex_conj_mul(xv, vx);
      w.real *= norm;
      w.imag *= norm;
      w.real += 1;
    }

// Intermediate result computing
    {
      complex_t *tmpxq = tmp0;
      complex_t *tmpxr = tmp1;
      complex_t *qk = sub_q;

      //SB_CONFIG(mulconj_config, mulconj_size);
      SB_CONTEXT(2);
      SB_DMA_READ(sub_q, 0, N * len * 8, 1, P_mulconj_Q);
      SB_DMA_READ(v, 0, 8 * len, N, P_mulconj_V);
      SB_CONST(P_mulconj_R, 0, i * len);
      SB_DMA_READ(r, 0, len * len * 8, 1, P_mulconj_R);

      SB_GARBAGE(P_mulconj_O1, i);
      SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, N);
      //SB_DMA_WRITE(P_mulconj_O0, 8, 8, N, tmpxq);
      SB_REPEAT_PORT(len);
      SB_XFER_RIGHT(P_mulconj_O0, P_finalize_compact_TMPQ, N);
      SB_DMA_WRITE(P_mulconj_O1, 8, 8, len, tmpxr);

      SB_WAIT_ALL();
    }

// Finalize the result
    {
      complex_t *tmpxq = tmp0, *nxt = sub_q, *qx = sub_q;

      SB_CONTEXT(4);
      //SB_CONFIG(finalize_compact_config, finalize_compact_size);
      SB_CONST(P_finalize_compact_W, *((uint64_t*)&w), N * len);
      SB_DMA_READ(qx, 8, 8, N * len, P_finalize_compact_Q);
      SB_DMA_READ(v, 0, 8 * len, N, P_finalize_compact_VQ); 
      SB_CONST(P_finalize_compact_TMPR, 0, i * len);
      SB_CONST(P_finalize_compact_VR, 0, i * len);
      SB_CONST(P_finalize_compact_R, 0, i * len);
      SB_GARBAGE(P_finalize_compact_O1, i * len);

      //SB_REPEAT_PORT(len);
      //SB_DMA_READ(tmpxq, 8, 8, N, P_finalize_compact_TMPQ);

      SB_2D_CONST(P_finalize_compact_MUX0, 0, 1, 1, len - 1, N);
      SB_DMA_WRITE(P_finalize_compact_O00, 8 * N, 8, N, Q + i);
      SB_DMA_WRITE(P_finalize_compact_O01, 8, 8, (len - 1) * N, nxt);

      {
        SB_DMA_READ(tmp1, 0, 8 * len, 1, P_finalize_compact_TMPR);
        SB_DMA_READ(r, len * 8, 8, len, P_finalize_compact_R);
        SB_CONST(P_finalize_compact_VR, *((uint64_t*)v), len);

        SB_DMA_WRITE(P_finalize_compact_O1, 8, 8, len, R + i * N + i);
      }


      SB_CONST(P_finalize_compact_TMPR, 0, len - 1);
      SB_CONST(P_finalize_compact_VR, 0, len - 1);
      SB_CONST(P_finalize_compact_R, 0, len - 1);
      SB_GARBAGE(P_finalize_compact_O1, len - 1);

      SB_DMA_READ(r + len + 1, 8 * len, 8 * (len - 1), len - 1, P_finalize_compact_R);
      SB_DMA_READ(v + 1, 0, 8 * (len - 1), len - 1, P_finalize_compact_VR);
      SB_DMA_WRITE(P_finalize_compact_O1, 8, 8, (len - 1) * (len - 1), r);


      SB_REPEAT_PORT(len - 1);
      SB_DMA_READ(tmp1 + 1, 8, 8, len - 1, P_finalize_compact_TMPR);

      SB_WAIT_ALL();
    }

  }
}
#undef h

