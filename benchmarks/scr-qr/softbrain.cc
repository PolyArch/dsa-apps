#include "qr.h"
#include <iostream>
#include "sim_timing.h"
#include "conjconj.h"
#include "mulconj.h"
#include "finalize.h"
#include "sb_insts.h"

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))

complex<float> _one(1, 0), _zero(0, 0);

void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int q = 0, tmp0 = 8192, r = 16384, tmp1 = 24576, v_addr = 32768;
  {
    SB_DMA_SCRATCH_LOAD(a, 8 * N, 8, N, 0);
    complex<float> v[N];
    {
      float norm = 0;
      for (int i = 0; i < N; ++i) {
        v[i] = a[i * N];
        norm += v[i].real() * v[i].real() + v[i].imag() * v[i].imag();
      }
      norm = sqrt(norm);
      float sign = sqrt(v->real() * v->real() + v->imag() * v->imag());
      *v = complex<float>(
        v->real() + v->real() / sign * norm,
        v->imag() + v->imag() / sign * norm
      );
    }

    {
      float norm = 0;
      for (int i = 0; i < N; ++i)
        norm += v[i].real() * v[i].real() + v[i].imag() * v[i].imag();

      norm = 1 / sqrt(norm);

      for (int i = 0; i < N; ++i) {
        v[i] = complex<float>(v[i].real() * norm, v[i].imag() * norm);
        //std::cout << v[i] << " ";
      }
      //std::cout << "\n";
    }

    SB_CONFIG(conjconj_config, conjconj_size);
    complex<float> w;
    {
      complex<float> xv = {0, 0}, vx = {0, 0};

      SB_WAIT_SCR_WR();
      SB_SCRATCH_READ(0, 8 * N, P_conjconj_A0);
      SB_DMA_READ(v, 8, 8, N, P_conjconj_B0);
      SB_DMA_READ(v, 8, 8, N, P_conjconj_A1);
      SB_SCRATCH_READ(0, 8 * N, P_conjconj_B1);
      SB_CONST(P_conjconj_reset, 2, N - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_RECV(P_conjconj_O0, xv);
      SB_RECV(P_conjconj_O1, vx);
      SB_WAIT_ALL();
      
      float norm = vx.real() * vx.real() + vx.imag() * vx.imag();
      w = complex<float>(
          -(xv.real() * vx.real() + xv.imag() * vx.imag()) / norm - 1,
          -(xv.imag() * vx.real() - xv.real() * vx.imag()) / norm
      );
      std::cout << w << "\n";
    }

    {
      SB_CONFIG(mulconj_config, mulconj_size);
      SB_DMA_READ(v, 0, 8 * N, N * N, P_mulconj_A0);
      SB_DMA_READ(v, 0, 8 * N, N * N, P_mulconj_A1);

      for (int y = 0; y < N; ++y) {
        SB_CONST(P_mulconj_B0, 0, y);
        SB_2D_CONST(P_mulconj_B0, 1, 1, 0, N, N - 1);
        SB_2D_CONST(P_mulconj_B0, 1, 1, 0, N - y - 1, 1);
        //SB_DMA_READ(q + y * N + i, 0, 8 * (len), len, P_mulconj_B0);
        for (int x = 0; x < N; ++x)
          SB_DMA_READ(a + x, 8 * N, 8, N, );
        //SB_DMA_READ(r + i * (N + 1), 8 * N, 8 * (len), len, P_mulconj_B1);
        complex_t *tmpx0 = tmp0 + y * N + i;
        complex_t *tmpx1 = tmp1 + y * N + i;
        SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, len);
        SB_DMA_WRITE(P_mulconj_O0, 8, 8, len, tmpx0);
        SB_DMA_WRITE(P_mulconj_O1, 8, 8, len, tmpx1);
      }
      SB_WAIT_ALL();
    }

  }

  /*
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
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, 1, &_norm);
      SB_GARBAGE(P_conjconj_O1, 1);
      SB_WAIT_ALL();

      float norm = sqrt(_norm.real);
      float sign = sqrt(v->real * v->real + v->imag * v->imag);
      v->real += v->real / sign * norm;
      v->imag += v->imag / sign * norm;
    }

    {
      complex_t *vp = v;

      complex_t _norm;

      SB_DMA_READ(v, 8, 8, len, P_conjconj_A0);
      SB_DMA_READ(v, 8, 8, len, P_conjconj_B0);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, 1, &_norm);
      SB_GARBAGE(P_conjconj_O1, 1);
      SB_WAIT_ALL();

      float norm = sqrt(_norm.real);
      for (j = i; j < N; ++j) {
        vp->real /= norm;
        vp->imag /= norm;
        ++vp;
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
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_DMA_WRITE(P_conjconj_O0, 8, 8, 1, &xv);
      SB_DMA_WRITE(P_conjconj_O1, 8, 8, 1, &vx);
      SB_WAIT_ALL();
      
      float norm = vx.real * vx.real + vx.imag * vx.imag;
      w.real = (xv.real * vx.real + xv.imag * vx.imag) / norm + 1;
      w.imag = (xv.imag * vx.real - xv.real * vx.imag) / norm;
      w.real = -w.real;
      w.imag = -w.imag;
    }

    {
      SB_CONFIG(mulconj_config, mulconj_size);
      SB_DMA_READ(v, 0, 8 * (len), (len) * N, P_mulconj_A0);
      SB_CONST(P_mulconj_A1, 0, i * len * len);
      SB_DMA_READ(v, 0, 8 * (len), (len) * (len), P_mulconj_A1);
      SB_CONST(P_mulconj_B1, 0, i * len * len);
      SB_GARBAGE(P_mulconj_O1, i * len);
      SB_DMA_WRITE(P_mulconj_O0, 8 * N, 8 * len, i, tmp0 + i);

      for (y = 0; y < i; ++y) {
        complex_t *tmpx0 = tmp0 + y * N + i;
        SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, len);
        SB_DMA_READ(q + y * N + i, 0, 8 * (len), len, P_mulconj_B0);
      }

      for (y = i; y < N; ++y) {
        SB_DMA_READ(q + y * N + i, 0, 8 * (len), len, P_mulconj_B0);
        SB_DMA_READ(r + i * (N + 1), 8 * N, 8 * (len), len, P_mulconj_B1);
        complex_t *tmpx0 = tmp0 + y * N + i;
        complex_t *tmpx1 = tmp1 + y * N + i;
        SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, len);
        SB_DMA_WRITE(P_mulconj_O0, 8, 8, len, tmpx0);
        SB_DMA_WRITE(P_mulconj_O1, 8, 8, len, tmpx1);
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
    SB_REPEAT_PORT(len);
    SB_DMA_READ(v, 8, 8, len, P_finalize_VY);
      SB_DMA_READ(tmp1 + i * (N + 1), 8 * N, 8 * len, len, P_finalize_A1);
    for (y = i; y < N; ++y) {
      complex_t *tmpx = tmp1 + y * N + i;
      SB_DMA_READ(r + i * N + y, 8 * N, 8, len, P_finalize_R);
      SB_DMA_WRITE(P_finalize_O1, 8 * N, 8, len, r + i * N + y);
    }
    SB_WAIT_ALL();

  }*/
  /*
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      R[j * N + i] = complex<float>(r[i * N + j].real, r[i * N + j].imag);
    }
  }
  for (i = 0; i < N * N; ++i) {
    Q[i] = complex<float>(q[i].real, q[i].imag);
  }*/
}
#undef h

