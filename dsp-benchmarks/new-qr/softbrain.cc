#include "qr.h"
#include <iostream>
#include "sim_timing.h"
#include "conjconj.h"
#include "mulconj.h"
#include "finalize.h"
#include "sb_insts.h"

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))

complex_t tmp0[N * N], tmp1[N * N];

void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int i, j, k, x, y;
  complex_t *q = new complex_t[N * N];
  complex_t *r = new complex_t[N * N];
  //complex_t *tmp = new complex_t[N * N];
  //complex<float> *tmp = new complex<float>[N * N];
  for (i = 0; i < N * N; ++i) {
    q[i] = i % N == i / N ? (complex_t){1, 0} : (complex_t){0, 0};
  }
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      r[j * N + i] = (complex_t){a[i * N + j].real(), a[i * N + j].imag()};
      //r[j * N + i] = a[i * N + j];
    }
  }

  for (i = 0; i < N; ++i) {
    int len = N - i;

    complex_t v[len];
    SB_CONFIG(conjconj_config, conjconj_size);
    complex_t *vp = v, *rp = r + i * (N + 1);
    for (j = i; j < N; ++j)
      *vp++ = *rp++;

    {
      complex_t _norm;

      SB_DMA_READ(v, 8, 8, len, P_conjconj_A0);
      SB_DMA_READ(v, 8, 8, len, P_conjconj_B0);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      //SB_CONST(P_conjconj_reset, 0, len - 1);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      //SB_GARBAGE(P_conjconj_O0, len - 1);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, 1, &_norm);
      //SB_GARBAGE(P_conjconj_O1, len);
      SB_GARBAGE(P_conjconj_O1, 1);
      SB_WAIT_ALL();

      float norm = sqrt(_norm.real);
      /*
      complex_t *vp = v, *rp = r + i * (N + 1);
      for (j = i; j < N; ++j) {
        *vp = *rp;
        norm += vp->real * vp->real + vp->imag * vp->imag;
        ++vp;
        ++rp;
      }
      norm = sqrt(norm);*/
      float sign = sqrt(v->real * v->real + v->imag * v->imag);
      v->real += v->real / sign * norm;
      v->imag += v->imag / sign * norm;
      //*v += *v / sign * norm;
    }

    {
      complex_t *vp = v;
      /*for (j = i; j < N; ++j) {
        norm += vp->real * vp->real + vp->imag * vp->imag;
        ++vp;
      }
      norm = sqrt(norm);
      vp = v;*/

      complex_t _norm;

      SB_DMA_READ(v, 8, 8, len, P_conjconj_A0);
      SB_DMA_READ(v, 8, 8, len, P_conjconj_B0);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      //SB_CONST(P_conjconj_reset, 0, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      //SB_GARBAGE(P_conjconj_O0, len - 1);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, 1, &_norm);
      //SB_GARBAGE(P_conjconj_O1, len);
      SB_GARBAGE(P_conjconj_O1, 1);
      SB_WAIT_ALL();

      float norm = sqrt(_norm.real);
      for (j = i; j < N; ++j) {
        vp->real /= norm;
        vp->imag /= norm;
        ++vp;
        //*vp++ /= norm;
      }
    }

    complex_t w;
    {
      complex_t xv = {0, 0}, vx = {0, 0};
      complex_t *rp = r + i * (N + 1),  *vp = v;

      SB_DMA_READ(rp, 8, 8, len, P_conjconj_A0);
      SB_DMA_READ(v, 8, 8, len, P_conjconj_B0);
      SB_DMA_READ(v, 8, 8, len, P_conjconj_A1);
      SB_DMA_READ(rp, 8, 8, len, P_conjconj_B1);
      //SB_CONST(P_conjconj_reset, 0, len - 1);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      //SB_GARBAGE(P_conjconj_O0, len - 1);
      //SB_GARBAGE(P_conjconj_O1, len - 1);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, 1, &xv);
      SB_DMA_WRITE(P_conjconj_O1, 8, 8, 1, &vx);
      SB_WAIT_ALL();
      
      /*
      for (j = i; j < N; ++j) {
        xv.real += rp->real * vp->real + rp->imag * vp->imag;
        xv.imag += rp->real * vp->imag - rp->imag * vp->real;
        vx.real += vp->real * rp->real + vp->imag * rp->imag;
        vx.imag += vp->real * rp->imag - vp->imag * rp->real;

        //xv += std::conj(r[i * N + j]) * v[j - i];
        //vx += std::conj(v[j - i]) * r[i * N + j];
        ++rp;
        ++vp;
      }*/
      float norm = vx.real * vx.real + vx.imag * vx.imag;
      w.real = (xv.real * vx.real + xv.imag * vx.imag) / norm + 1;
      w.imag = (xv.imag * vx.real - xv.real * vx.imag) / norm;
      w.real = -w.real;
      w.imag = -w.imag;
      //w = one + xv / vx;
    }

    {
      SB_CONFIG(mulconj_config, mulconj_size);
      SB_DMA_READ(v, 0, 8 * (len), (len) * N, P_mulconj_A0);
      SB_CONST(P_mulconj_A1, 0, i * len * len);
      SB_DMA_READ(v, 0, 8 * (len), (len) * (len), P_mulconj_A1);
      SB_CONST(P_mulconj_B1, 0, i * len * len);
      //SB_GARBAGE(P_mulconj_O1, i * len * len);
      SB_GARBAGE(P_mulconj_O1, i * len);
      SB_DMA_WRITE(P_mulconj_O0, 8 * N, 8 * len, i, tmp0 + i);

      for (y = 0; y < i; ++y) {
        complex_t *tmpx0 = tmp0 + y * N + i;
        SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, len);
        SB_DMA_READ(q + y * N + i, 0, 8 * (len), len, P_mulconj_B0);
        //complex_t *tmpx1 = tmp1 + y * N + i;
        //SB_DMA_READ(q + y * N + i, 0, 8 * (len), len, P_mulconj_B0);
        //SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, len);
        //SB_DMA_WRITE(P_mulconj_O0, 0, 8 * (N - i), 1, tmpx0);
        //for (x = i; x < N; ++x) {
          //SB_CONST(P_mulconj_reset, 0, len - 1);
          //SB_CONST(P_mulconj_reset, 1, 1);
          //SB_GARBAGE(P_mulconj_O0, len - 1);
          //SB_DMA_WRITE(P_mulconj_O0, 8, 8, 1, tmpx0);
          //++tmpx0;
        //}
      }

      for (y = i; y < N; ++y) {
        SB_DMA_READ(q + y * N + i, 0, 8 * (len), len, P_mulconj_B0);
        SB_DMA_READ(r + i * (N + 1), 8 * N, 8 * (len), len, P_mulconj_B1);
        complex_t *tmpx0 = tmp0 + y * N + i;
        complex_t *tmpx1 = tmp1 + y * N + i;
        SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, len);
        SB_DMA_WRITE(P_mulconj_O0, 8, 8, len, tmpx0);
        SB_DMA_WRITE(P_mulconj_O1, 8, 8, len, tmpx1);
        /*for (x = i; x < N; ++x) {
          SB_CONST(P_mulconj_reset, 0, len - 1);
          SB_CONST(P_mulconj_reset, 1, 1);
          SB_GARBAGE(P_mulconj_O0, len - 1);
          SB_DMA_WRITE(P_mulconj_O0, 8, 8, 1, tmpx0);
          ++tmpx0;
          SB_DMA_READ(r + i + x * N, 0, 8 * (len), 1, P_mulconj_B1);
          SB_GARBAGE(P_mulconj_O1, len - 1);
          SB_DMA_WRITE(P_mulconj_O1, 8, 8, 1, tmpx1);
          ++tmpx1;
        }*/
      }
      SB_WAIT_ALL();
    }

    SB_CONFIG(finalize_config, finalize_size);
    SB_DMA_READ(v, 0, 8 * (len), N, P_finalize_A0);
    SB_DMA_READ(tmp0 + i, N * 8, 8 * (len), N, P_finalize_B0);
    SB_CONST(P_finalize_W, *((uint64_t*)&w), (len) * N);
    SB_DMA_READ(q + i, 8 * N, 8 * (len), N, P_finalize_Q);
    SB_DMA_WRITE(P_finalize_O0, 8 * N, 8 * (len), N, q + i);

    SB_CONST(P_finalize_A1, 0, len * i);
    SB_CONST(P_finalize_R, 0, len * i);
    SB_CONST(P_finalize_VY, 0, len * i);
    SB_GARBAGE(P_finalize_O1, len * i);
    //complex_t *vy = v;
    SB_REPEAT_PORT(len);
    SB_DMA_READ(v, 8, 8, len, P_finalize_VY);
      SB_DMA_READ(tmp1 + i * (N + 1), 8 * N, 8 * len, len, P_finalize_A1);
    for (y = i; y < N; ++y) {
      complex_t *tmpx = tmp1 + y * N + i;
      //SB_DMA_READ(tmpx, 0, 8 * (len), 1, P_finalize_A1);
      //SB_CONST(P_finalize_VY, *((uint64_t*)vy), len);
      SB_DMA_READ(r + i * N + y, 8 * N, 8, len, P_finalize_R);
      SB_DMA_WRITE(P_finalize_O1, 8 * N, 8, len, r + i * N + y);

      /*for (x = i; x < N; ++x) {
        complex_t val = {
            tmpx->real * w.real - tmpx->imag * w.imag,
            tmpx->real * w.imag + tmpx->imag * w.real};
        complex_t delta = {
            val.real * vy->real - val.imag * vy->imag,
            val.real * vy->imag + val.imag * vy->real};
        r[x * N + y].real += delta.real;
        r[x * N + y].imag += delta.imag;
        ++tmpx;
        //r[x * N + y] -= tmp[y * N + x] * w * v[y - i];
      }*/
      //++vy;
    }
    SB_WAIT_ALL();

  }
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      R[j * N + i] = complex<float>(r[i * N + j].real, r[i * N + j].imag);
    }
  }
  for (i = 0; i < N * N; ++i) {
    Q[i] = complex<float>(q[i].real, q[i].imag);
  }
}
#undef h

