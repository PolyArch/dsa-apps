#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"

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
    complex_t v[len];
    {
      float norm = 0;
      complex_t *vp = v, *rp = r;
      for (j = i; j < N; ++j) {
        *vp = *rp;
        norm += complex_norm(*vp); //vp->real * vp->real + vp->imag * vp->imag;
        ++vp;
        ++rp;
      }
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
// Household Vector Done

    {
      complex_t *tmpxq = tmp0;
      complex_t *tmpxr = tmp1;
      complex_t *qk = sub_q;

      for (y = 0; y < i; ++y) {
        tmpxq->real = tmpxq->imag = 0;
        complex_t *vk = v;
        for (k = 0; k < len; ++k) {
          complex_t delta = complex_mul(*vk, *qk);
          *tmpxq = complex_add(*tmpxq, delta);
          ++vk;
          ++qk;
        }
        ++tmpxq;
      }

      for (x = 0; x < len; ++x) {
        tmpxq->real = tmpxq->imag = tmpxr->real = tmpxr->imag = 0;
        complex_t *vk = v;
        complex_t *rk = r + x * len;
        for (k = i; k < N; ++k) {
          complex_t delta = complex_conj_mul(*vk, *rk);
          *tmpxr = complex_add(*tmpxr, delta);

          delta = complex_mul(*vk, *qk);
          *tmpxq = complex_add(*tmpxq, delta);

          ++qk;
          ++vk;
          ++rk;
        }
        ++tmpxr;
        ++tmpxq;
      }
    }

    {
      complex_t *tmpxq = tmp0, *nxt = sub_q, *qx = sub_q;
      for (y = 0; y < i; ++y) {
        complex_t *vx = v;
        {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          complex_t delta = complex_mul(val, w);
          Q[y * N + i] = complex<float>(qx->real - delta.real, qx->imag - delta.imag);
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * w;
        }
        for (x = 1; x < len; ++x) {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          complex_t delta = complex_mul(val, w);
          *nxt = complex_sub(*qx, delta);
          ++nxt;
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * w;
        }
        tmpxq++;
      }

      {
        complex_t *vy = v;
        complex_t *tmpxr = tmp1;
        complex_t *read = r;
        {
          complex_t val = complex_mul(*tmpxr, w);
          complex_t delta = complex_mul(val, *vy);
          R[i * N + i] = complex<float>(
              read[0].real - delta.real,
              read[0].imag - delta.imag
          );
          ++tmpxr;
          //r[x * N + y] -= tmp[y * N + x] * w * v[y - i];
        }
        complex_t *vx = v;
        {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          complex_t delta = complex_mul(val, w);
          Q[i * N + i] = complex<float>(qx->real - delta.real, qx->imag - delta.imag);
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * w;
        }
        for (x = i + 1; x < N; ++x) {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          complex_t delta = complex_mul(val, w);
          *nxt = complex_sub(*qx, delta);
          ++nxt;
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * w;

          val = complex_mul(*tmpxr, w);
          delta = complex_mul(val, *vy);
          R[i * N + x] = complex<float>(
              read[len * (x - i)].real - delta.real,
              read[len * (x - i)].imag - delta.imag
          );
          ++tmpxr;
          //r[x * N + y] -= tmp[y * N + x] * w * v[y - i];
        }
        ++vy;
        tmpxq++;
      }
      complex_t *tmpyr = tmp1 + 1;
      complex_t *write = r;
      for (y = i + 1; y < N; ++y) {
        complex_t *vx = v;
        {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          complex_t delta = complex_mul(val, w);
          Q[y * N + i] = complex<float>(qx->real - delta.real, qx->imag - delta.imag);
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * w;
        }

        complex_t *read = r + (y - i) * len + 1;
        complex_t *tmpxr = tmp1 + 1;
        for (x = i + 1; x < N; ++x) {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          complex_t delta = complex_mul(val, w);
          *nxt = complex_sub(*qx, delta);
          ++nxt;
          ++qx;

          val = complex_mul(*tmpyr, w);
          delta = complex_mul(val, *vx);
          *write = complex_sub(*read, delta);
          ++write;
          ++read;
          ++vx;

          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * w;
        }
        //++vy;
        tmpxq++;
        tmpyr++;
      }
    }

    //std::swap(r, _r);
  }
}
#undef h

