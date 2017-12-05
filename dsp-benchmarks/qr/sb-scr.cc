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

complex_t sub_q[N * N];

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int i, j, k, x, y;
  for (i = 0; i < N * N; ++i) {
    sub_q[i] = i % N == i / N ? (complex_t){1, 0} : (complex_t){0, 0};
  }

  int r = 24 * N;
  //complex_t *r = rbuffer0;
  for (i = 0; i < N; ++i) {
    SB_DMA_SCRATCH_LOAD(a + i, 8 * N, 8, N, r + i * 8 * N);
  }
  SB_WAIT_ALL();

  /*for (i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      complex_t temp;
      SB_SCRATCH_DMA_STORE(r + i * 8 * N + j * 8, 0, 8, 1, &temp);
      SB_WAIT_ALL();
      printf("%f %f\n", temp.real, temp.imag);
    }
    puts("");
  }*/

  for (i = 0; i < N; ++i) {
    int len = N - i;
// Household Vector
    complex_t head;
    {
      float norm = 0;

      SB_CONFIG(conjconj_config, conjconj_size);
      SB_SCRATCH_READ(r, 8 * len, P_conjconj_A0);
      SB_SCRATCH_READ(r, 8 * len, P_conjconj_B0);
      //SB_DMA_READ(r, 8, 8, len, P_conjconj_A0);
      //SB_DMA_READ(r, 8, 8, len, P_conjconj_B0);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      SB_CONST(P_conjconj_reset, 2, len - 1);
      SB_CONST(P_conjconj_reset, 1, 1);
      SB_CONST(P_conjconj_sqrt, 0, len - 1);
      SB_CONST(P_conjconj_sqrt, 1, 1);

      SB_GARBAGE(P_conjconj_O0, 1);
      SB_GARBAGE(P_conjconj_O1, 1);
      SB_RECV(P_conjconj_O2, norm);

      //complex_t temp = *r;
      complex_t temp;
      SB_SCRATCH_DMA_STORE(r, 0, 8, 1, &temp);
      SB_WAIT_ALL();
      float sign = sqrt(temp.real * temp.real + temp.imag * temp.imag);
      head.real = temp.real + temp.real / sign * norm;
      head.imag = temp.imag + temp.imag / sign * norm;
      //std::cout << temp.real << ", " << temp.imag << "\n";
      //std::cout << head.real << ", " << head.imag << "\n";
    }

    {
      float norm = 0;
      SB_CONST(P_conjconj_A0, *((uint64_t *) &head), 1);
      SB_CONST(P_conjconj_B0, *((uint64_t *) &head), 1);
      //SB_DMA_READ(r + 1, 8, 8, len - 1, P_conjconj_A0);
      //SB_DMA_READ(r + 1, 8, 8, len - 1, P_conjconj_B0);
      SB_SCRATCH_READ(r + 8, 8 * (len - 1), P_conjconj_A0);
      SB_SCRATCH_READ(r + 8, 8 * (len - 1), P_conjconj_B0);
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
      //SB_DMA_READ(r + 1, 8, 8, len - 1, P_conjconj_B0);
      SB_SCRATCH_READ(r + 8, 8 * (len - 1), P_conjconj_B0);
      SB_CONST(P_conjconj_A0, ri.b, len);
      SB_CONST(P_conjconj_A1, 0, len);
      SB_CONST(P_conjconj_B1, 0, len);
      SB_CONST(P_conjconj_reset, 1, len);
      SB_CONST(P_conjconj_sqrt, 0, len);
      SB_SCR_WRITE(P_conjconj_O0, 8 * len, 0);
      SB_GARBAGE(P_conjconj_O1, len);

      SB_WAIT_ALL();

    }

    complex_t w;
    {
      complex_t xv = {0, 0}, vx = {0, 0};

      //SB_DMA_READ(r, 8, 8, len, P_conjconj_A0);
      SB_SCRATCH_READ(r, 8 * len, P_conjconj_A0);
      SB_SCRATCH_READ(0, 8 * len, P_conjconj_B0);
      SB_SCRATCH_READ(0, 8 * len, P_conjconj_A1);
      //SB_DMA_READ(r, 8, 8, len, P_conjconj_B1);
      SB_SCRATCH_READ(r, 8 * len, P_conjconj_B1);
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

    int tmp0 = 8 * len;
    int tmp1 = 8 * len + 8 * N;
// Intermediate result computing
    {
      complex_t *qk = sub_q;

      SB_CONFIG(mulconj_config, mulconj_size);
      SB_DMA_READ(sub_q, 0, N * len * 8, 1, P_mulconj_Q);
      SB_SCR_PORT_STREAM(0, 0, 8 * len, N, P_mulconj_V);
      SB_CONST(P_mulconj_R, 0, i * len);
      //SB_DMA_READ(r, 0, len * len * 8, 1, P_mulconj_R);
      SB_SCRATCH_READ(r, 8 * len * len, P_mulconj_R);

      SB_GARBAGE(P_mulconj_O1, i);
      SB_2D_CONST(P_mulconj_reset, 2, len - 1, 1, 1, N);
      SB_SCR_WRITE(P_mulconj_O0, 8 * N, tmp0);
      SB_SCR_WRITE(P_mulconj_O1, 8 * len, tmp1);

      SB_WAIT_ALL();
    }

// Finalize the result
    {
      complex_t *nxt = sub_q, *qx = sub_q;

      SB_CONFIG(finalize_compact_config, finalize_compact_size);
      SB_CONST(P_finalize_compact_W, *((uint64_t*)&w), N * len);
      SB_DMA_READ(qx, 8, 8, N * len, P_finalize_compact_Q);
      SB_SCR_PORT_STREAM(1, 0, 8 * len, N, P_finalize_compact_VQ);
      SB_CONST(P_finalize_compact_TMPR, 0, i * len);
      SB_CONST(P_finalize_compact_VR, 0, i * len);
      SB_CONST(P_finalize_compact_R, 0, i * len);
      SB_GARBAGE(P_finalize_compact_O1, i * len);

      SB_REPEAT_PORT(len);
      SB_SCRATCH_READ(tmp0, 8 * N, P_finalize_compact_TMPQ);

      SB_2D_CONST(P_finalize_compact_MUX0, 0, 1, 1, len - 1, N);
      SB_DMA_WRITE(P_finalize_compact_O00, 8 * N, 8, N, Q + i);
      SB_DMA_WRITE(P_finalize_compact_O01, 8, 8, (len - 1) * N, nxt);

      {
        SB_SCRATCH_READ(tmp1, 8 * len, P_finalize_compact_TMPR);
        //SB_DMA_READ(r, len * 8, 8, len, P_finalize_compact_R);
        SB_SCR_PORT_STREAM(r, len * 8, 8, len, P_finalize_compact_R);
        SB_SCR_PORT_STREAM(0, 0, 8, len, P_finalize_compact_VR);

        SB_DMA_WRITE(P_finalize_compact_O1, 8, 8, len, R + i * N + i);
      }


      SB_CONST(P_finalize_compact_TMPR, 0, len - 1);
      SB_CONST(P_finalize_compact_VR, 0, len - 1);
      SB_CONST(P_finalize_compact_R, 0, len - 1);
      SB_GARBAGE(P_finalize_compact_O1, len - 1);

      //SB_DMA_READ(r + len + 1, 8 * len, 8 * (len - 1), len - 1, P_finalize_compact_R);
      SB_SCR_PORT_STREAM(r + (len + 1) * 8, 8 * len, 8 * (len - 1), len - 1, P_finalize_compact_R);
      SB_SCR_PORT_STREAM(8, 0, 8 * (len - 1), len - 1, P_finalize_compact_VR);
      //SB_DMA_WRITE(P_finalize_compact_O1, 8, 8, (len - 1) * (len - 1), r);
      SB_SCR_WRITE(P_finalize_compact_O1, 8 * (len - 1) * (len - 1), r);


      SB_REPEAT_PORT(len - 1);
      SB_SCRATCH_READ(tmp1 + 8, 8 * (len - 1), P_finalize_compact_TMPR);

      SB_WAIT_ALL();
    }
  }
}
#undef h

