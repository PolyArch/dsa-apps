#include "qr.h"
#include <iostream>
#include "sim_timing.h"

complex_t tmp0[N * N];
complex_t tmp1[N * N];
complex_t sub_q[N * N];
//double buffering
complex_t rbuffer0[N * N];
complex_t rbuffer1[N * N];

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int i, j, k, x, y;
  complex_t *r = rbuffer0, *_r = rbuffer1;
  for (i = 0; i < N * N; ++i) {
    sub_q[i] = i % N == i / N ? (complex_t){1, 0} : (complex_t){0, 0};
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
    {
      float norm = 0;
      complex_t *vp = v, *rp = r + i * (N + 1);
      for (j = i; j < N; ++j) {
        *vp = *rp;
        norm += vp->real * vp->real + vp->imag * vp->imag;
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
      complex_t *rp = r + i * (N + 1),  *vp = v;
      for (j = i; j < N; ++j) {
        xv.real += rp->real * vp->real + rp->imag * vp->imag;
        xv.imag += rp->real * vp->imag - rp->imag * vp->real;
        vx.real += vp->real * rp->real + vp->imag * rp->imag;
        vx.imag += vp->real * rp->imag - vp->imag * rp->real;

        //xv += std::conj(r[i * N + j]) * v[j - i];
        //vx += std::conj(v[j - i]) * r[i * N + j];
        ++rp;
        ++vp;
      }
      float norm = vx.real * vx.real + vx.imag * vx.imag;
      w.real = (xv.real * vx.real + xv.imag * vx.imag) / norm + 1;
      w.imag = (xv.imag * vx.real - xv.real * vx.imag) / norm;
      //w = one + xv / vx;
    }

    {
      complex_t *tmpx = tmp0;
      for (y = 0; y < N; ++y) {
        for (x = i; x < N; ++x) {
          tmpx->real = tmpx->imag = 0;
          complex_t *vk = v, *qk = sub_q + y * len;
          for (k = i; k < N; ++k) {
            tmpx->real += vk->real * qk->real - vk->imag * qk->imag;
            tmpx->imag += vk->real * qk->imag + vk->imag * qk->real;
            //tmp[y * N + x] += v[k - i] * q[y * N + k];
            ++vk;
            ++qk;
          }
          ++tmpx;
        }
      }
    }

    {
      complex_t *tmpx = tmp0, *nxt = sub_q, *qx = sub_q;
      for (y = 0; y < N; ++y) {
        complex_t *vx = v;
        for (x = i; x < N; ++x) {
          complex_t val = {
              vx->real * tmpx->real + vx->imag * tmpx->imag,
              vx->real * tmpx->imag - vx->imag * tmpx->real};
          complex_t delta = {
              val.real * w.real - val.imag * w.imag,
              val.real * w.imag + val.imag * w.real
          };
          if (x == i) {
            Q[y * N + i] = complex<float>(qx->real - delta.real, qx->imag - delta.imag);
            //std::cout << Q[y * N + i] << "\n";
          } else {
            nxt->real = qx->real - delta.real;
            nxt->imag = qx->imag - delta.imag;
            ++nxt;
          }
          ++qx;
          ++tmpx;
          ++vx;
          //q[y * N + x] -= tmp[y * N + x] * std::conj(v[x - i]) * w;
        }
      }
    }

    {
      complex_t *tmpx = tmp1;
      for (y = i; y < N; ++y) {
        for (x = i; x < N; ++x) {
          tmpx->real = tmpx->imag = 0;
          complex_t *rk = r + x * N + i, *vk = v;
          for (k = i; k < N; ++k) {
            //tmp[y * N + x] += r[x * N + k] * std::conj(v[k - i]);
            complex_t delta = {
                vk->real * rk->real + vk->imag * rk->imag,
                vk->real * rk->imag - vk->imag * rk->real};
            tmpx->real += delta.real;
            tmpx->imag += delta.imag;
            ++vk;
            ++rk;
          }
          ++tmpx;
        }
      }
    }
    complex_t *vy = v;
    complex_t *tmpx = tmp1;
    for (y = i; y < N; ++y) {
      complex_t *read = r + y + i * N;
      for (x = i; x < N; ++x) {
        complex_t val = {
            tmpx->real * w.real - tmpx->imag * w.imag,
            tmpx->real * w.imag + tmpx->imag * w.real};
        complex_t delta = {
            val.real * vy->real - val.imag * vy->imag,
            val.real * vy->imag + val.imag * vy->real};
        //r[x * N + y].real -= delta.real;
        //r[x * N + y].imag -= delta.imag;
        if (y == i) {
          R[y * N + x] = complex<float>(
              read->real - delta.real,
              read->imag - delta.imag
          );
        } else {
          read->real -= delta.real;
          read->imag -= delta.imag;
        }
        read += N;
        ++tmpx;
        //r[x * N + y] -= tmp[y * N + x] * w * v[y - i];
      }
      ++vy;
    }

  }
}
#undef h

