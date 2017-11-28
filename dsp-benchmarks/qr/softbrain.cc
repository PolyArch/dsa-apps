#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "mulconj.h"
#include "finalize.h"

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
complex_t rbuffer1[N * N];

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int i, j, k, x, y;
  for (i = 0; i < N * N; ++i) {
    sub_q[i] = i % N == i / N ? (complex_t){1, 0} : (complex_t){0, 0};
  }

  complex_t *r = rbuffer0, *_r = rbuffer1;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      r[j * N + i] = (complex_t){a[i * N + j].real(), a[i * N + j].imag()};
      //r[j * N + i] = a[i * N + j];
    }
  }

  for (i = 0; i < N; ++i) {
    int len = N - i;
// Household Vector
    complex_t v[len];
    {
      float norm = 0;
      complex_t *vp = v, *rp = r;
      for (j = i; j < N; ++j) {
        *vp = *rp;
        norm += complex_norm(*vp); //vp->real * vp->real + vp->imag * vp->imag;
        //std::cout << complex<float>(vp->real, vp->imag) << " ";
        ++vp;
        ++rp;
      }
      //std::cout << "\n";
      norm = sqrt(norm);
      float sign = sqrt(v->real * v->real + v->imag * v->imag);
      v->real += v->real / sign * norm;
      v->imag += v->imag / sign * norm;
      //*v += *v / sign * norm;
    }

    {
      float norm = 0;
      complex_t *vp = v;
      for (j = i; j < N; ++j) {
        norm += vp->real * vp->real + vp->imag * vp->imag;
        ++vp;
      }
      norm = 1. / sqrt(norm);
      vp = v;
      for (j = i; j < N; ++j) {
        vp->real *= norm;
        vp->imag *= norm;
        ++vp;
        //*vp++ /= norm;
      }
    }

    complex_t w;
    {
      complex_t xv = {0, 0}, vx = {0, 0};
      complex_t *rp = r,  *vp = v;
      for (j = i; j < N; ++j) {
        complex_t delta = complex_conj_mul(*rp, *vp);
        xv = complex_add(xv, delta);
        delta = complex_conj_mul(*vp, *rp);
        vx = complex_add(vx, delta);
        ++rp;
        ++vp;
      }
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

      SB_CONFIG(mulconj_config, mulconj_size);
      SB_DMA_READ(sub_q, 0, N * len * 8, 1, P_mulconj_Q);
      SB_DMA_READ(v, 0, 8 * len, N, P_mulconj_V);
      SB_CONST(P_mulconj_R, 0, i * len);
      SB_DMA_READ(r, 0, len * len * 8, 1, P_mulconj_R);

      SB_GARBAGE(P_mulconj_O1, len * i);
      for (y = 0; y < i; ++y) {
        SB_CONST(P_mulconj_reset, 0, len - 1);
        SB_CONST(P_mulconj_reset, 1, 1);
        SB_GARBAGE(P_mulconj_O0, len - 1);
        SB_DMA_WRITE(P_mulconj_O0, 0, 8, 1, tmpxq++);
      }

      for (x = 0; x < len; ++x) {
        SB_CONST(P_mulconj_reset, 0, len - 1);
        SB_CONST(P_mulconj_reset, 1, 1);
        SB_GARBAGE(P_mulconj_O0, len - 1);
        SB_DMA_WRITE(P_mulconj_O0, 0, 8, 1, tmpxq++);
        SB_GARBAGE(P_mulconj_O1, len - 1);
        SB_DMA_WRITE(P_mulconj_O1, 0, 8, 1, tmpxr++);
      }

      SB_WAIT_ALL();
    }

// Finalize the result
    {
      complex_t *tmpxq = tmp0, *nxt = sub_q, *qx = sub_q;

      SB_CONFIG(finalize_config, finalize_size);
      SB_CONST(P_finalize_W, *((uint64_t*)&w), i * len);
      SB_DMA_READ(qx, 8, 8, i * len, P_finalize_Q);
      SB_DMA_READ(v, 0, 8 * len, i, P_finalize_VQ);

      SB_CONST(P_finalize_TMPR, 0, i * len);
      SB_CONST(P_finalize_VR, 0, i * len);
      SB_CONST(P_finalize_R, 0, i * len);
      SB_GARBAGE(P_finalize_O1, i * len);

      for (y = 0; y < i; ++y) {
        SB_CONST(P_finalize_TMPQ, *((uint64_t*)tmpxq), len);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, 1, Q + y * N + i);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, len - 1, nxt);
        tmpxq++;
        nxt += len - 1;
      }
      qx += i * len;


      {
        SB_DMA_READ(tmp1, 0, 8 * len, 1, P_finalize_TMPR);
        SB_DMA_READ(r, len * 8, 8, len, P_finalize_R);
        SB_CONST(P_finalize_VR, *((uint64_t*)v), len);

        SB_DMA_READ(qx, 8, 8, len, P_finalize_Q);
        SB_CONST(P_finalize_TMPQ, *((uint64_t*)tmpxq), len);
        SB_DMA_READ(v, 8, 8, len, P_finalize_VQ);

        SB_CONST(P_finalize_W, *((uint64_t*)&w), len);

        SB_DMA_WRITE(P_finalize_O1, 8, 8, len, R + i * N + i);
        SB_DMA_WRITE(P_finalize_O0, 0, 8, 1, Q + i * N + i);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, len - 1, nxt);

        ++tmpxq;
        qx += len;
        nxt += len - 1;

      }

 
      SB_CONST(P_finalize_W, *((uint64_t*)&w), (len - 1) * len);

      SB_DMA_READ(v, 0, 8 * len, len - 1, P_finalize_VQ);
      SB_DMA_READ(qx, 8, 8, len * (len - 1), P_finalize_Q);

      SB_CONST(P_finalize_TMPR, 0, len - 1);
      SB_CONST(P_finalize_VR, 0, len - 1);
      SB_CONST(P_finalize_R, 0, len - 1);
      SB_GARBAGE(P_finalize_O1, len - 1);

      SB_DMA_READ(r + len + 1, 8 * len, 8 * (len - 1), len - 1, P_finalize_R);
      SB_DMA_READ(v + 1, 0, 8 * (len - 1), len - 1, P_finalize_VR);
      SB_DMA_WRITE(P_finalize_O1, 8, 8, (len - 1) * (len - 1), r);

      for (y = i + 1; y < N; ++y) {
        SB_CONST(P_finalize_TMPQ, *((uint64_t*)tmpxq), len);
        SB_CONST(P_finalize_TMPR, *((uint64_t*)tmp1 + y - i), len - 1);
        SB_DMA_WRITE(P_finalize_O0, 0, 8, 1, Q + y * N + i);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, len - 1, nxt);
        ++tmpxq;
        nxt += len - 1;
      }

      SB_WAIT_ALL();
    }

  }
}
#undef h

