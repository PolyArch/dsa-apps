#include "qr.h"
#include <iostream>
#include <algorithm>

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
      complex_t *vp = v, *rp = r;
      *vp = *rp;
      float norm0 = complex_norm(*vp), norm1 = 0;
      ++vp; ++rp;
      for (j = i + 1; j < N; ++j) {
        *vp = *rp;
        norm1 += complex_norm(*vp); //vp->real * vp->real + vp->imag * vp->imag;
        ++vp;
        ++rp;
      }
      //norm0 += norm1;
      //float sign = v->real * v->real + v->imag * v->imag;
      float rate = 1 + sqrt(1 + norm1 / norm0);
      //printf("%f %f %f\n", norm0, norm1, rate);
      v->real *= rate;
      v->imag *= rate;

      norm1 += complex_norm(*v);
      //printf("%f\n", sqrt(norm1));
      norm1 = 1. / sqrt(norm1);
      vp = v;
      for (j = i; j < N; ++j) {
        vp->real *= norm1;
        vp->imag *= norm1;
        ++vp;
        //*vp++ /= norm;
      }
      /*for (int i = 0; i < len; ++i) {
        std::cout << "(" << v[i].real << "," << v[i].imag << ") ";
      }
      std::cout << "\n";*/

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
        tmpxq->real *= 2;
        tmpxq->imag *= 2;
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
        tmpxr->real *= 2;
        tmpxr->imag *= 2;
        ++tmpxr;

        tmpxq->real *= 2;
        tmpxq->imag *= 2;
        ++tmpxq;
      }
    }

    {
      complex_t *tmpxq = tmp0, *nxt = sub_q, *qx = sub_q;
      for (y = 0; y < i; ++y) {
        complex_t *vx = v;
        {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          Q[y * N + i] = complex<float>(qx->real - val.real, qx->imag - val.imag);
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * 2;
        }
        for (x = 1; x < len; ++x) {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          *nxt = complex_sub(*qx, val);
          ++nxt;
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * 2;
        }
        tmpxq++;
      }

      {
        complex_t *vy = v;
        complex_t *tmpxr = tmp1;
        complex_t *read = r;
        {
          complex_t delta = complex_mul(*tmpxr, *vy);
          R[i * N + i] = complex<float>(
              read[0].real - delta.real,
              read[0].imag - delta.imag
          );
          ++tmpxr;
          //r[x * N + y] -= tmp[y * N + x] * 2 * v[y - i];
        }
        complex_t *vx = v;
        {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          Q[i * N + i] = complex<float>(qx->real - val.real, qx->imag - val.imag);
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * 2;
        }
        for (x = i + 1; x < N; ++x) {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          *nxt = complex_sub(*qx, val);
          ++nxt;
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * 2;

          val = complex_mul(*tmpxr, *vy);
          R[i * N + x] = complex<float>(
              read[len * (x - i)].real - val.real,
              read[len * (x - i)].imag - val.imag
          );
          ++tmpxr;
          //r[x * N + y] -= tmp[y * N + x] * 2 * v[y - i];
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
          Q[y * N + i] = complex<float>(qx->real - val.real, qx->imag - val.imag);
          ++qx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * 2;
        }

        complex_t *read = r + (y - i) * len + 1;
        complex_t *tmpxr = tmp1 + 1;
        for (x = i + 1; x < N; ++x) {
          complex_t val = complex_conj_mul(*vx, *tmpxq);
          *nxt = complex_sub(*qx, val);
          ++nxt;
          ++qx;

          complex_t delta = complex_mul(*tmpyr, *vx);
          *write = complex_sub(*read, delta);
          ++write;
          ++read;
          ++vx;

          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * 2;
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

