#include "qr.h"
#include <iostream>
#include <algorithm>
#include "sim_timing.h"
#include "sb_insts.h"

#include "dot.dfg.h"
#include "mulconj.dfg.h"
#include "finalize.dfg.h"

#define complex_mul(a, b) (complex_t) { \
  (a).real * (b).real - (a).imag * (b).imag, \
  (a).real * (b).imag + (a).imag * (b).real }

#define complex_conj_mul(a, b) (complex_t) { \
  (a).real * (b).real + (a).imag * (b).imag, \
  (a).real * (b).imag - (a).imag * (b).real }

#define complex_add(a, b) (complex_t) { (a).real + (b).real, (a).imag + (b).imag }

#define complex_sub(a, b) (complex_t) { (a).real - (b).real, (a).imag - (b).imag }

#define complex_norm(a) ((a).real * (a).real + (a).imag * (a).imag)

complex_t tmp0[_N_ * _N_];
complex_t tmp1[_N_ * _N_];
complex_t sub_q[_N_ * _N_];
complex_t rbuffer0[_N_ * _N_];

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(complex<float> *a, complex<float> *Q, complex<float> *R) {
  int i, j, k, x, y;

  complex_t *r = rbuffer0;
  /*for (i = 0; i < _N_; ++i) {
    for (j = 0; j < _N_; ++j) {
      r[j * _N_ + i] = (complex_t){a[i * _N_ + j].real(), a[i * _N_ + j].imag()};
      //r[j * _N_ + i] = a[i * _N_ + j];
    }
  }*/

  for (i = 0; i < _N_; ++i) {
    int len = _N_ - i;
    complex_t v[len];
    {
      complex_t norm1 = {0, 0};

      SB_CONFIG(dot_config, dot_size);
      if (len > 1) {
        if (i) {
          SB_DMA_READ(r + 1, 8, 8, len - 1, P_dot_A);
          SB_DMA_READ(r + 1, 8, 8, len - 1, P_dot_B);
        } else {
          SB_DMA_READ(a + _N_, 8 * _N_, 8, len - 1, P_dot_A);
          SB_DMA_READ(a + _N_, 8 * _N_, 8, len - 1, P_dot_B);
        }
        SB_CONST(P_dot_reset, 0, len - 2);
        SB_CONST(P_dot_reset, 1, 1);
        SB_GARBAGE(P_dot_O, len - 2);
        SB_DMA_WRITE(P_dot_O, 8, 8, 1, &norm1);
      }

      if (i)
        *v = *r;
      else
        *v = (complex_t) {a->real(), a->imag()};
      float norm0 = complex_norm(*v);

      if (len > 1) {
        SB_WAIT_ALL();
      }


      float rate = 1 + sqrt(1 + norm1.real / norm0);
      v->real *= rate;
      v->imag *= rate;

      norm1.real += complex_norm(*v);
      norm1.real = 1. / sqrt(norm1.real);


      if (len > 1) {
        SB_CONST(P_dot_A, *((uint64_t*) &norm1), len - 1);
        if (i) {
          SB_DMA_READ(r + 1, 8, 8, len - 1, P_dot_B);
        } else {
          SB_DMA_READ(a + _N_, 8 * _N_, 8, len - 1, P_dot_B);
        }
        SB_CONST(P_dot_reset, 1, len - 1);
        SB_DMA_WRITE(P_dot_O, 8, 8, len - 1, v + 1);

        v->real *= norm1.real;
        v->imag *= norm1.real;

        SB_WAIT_ALL();
      } else {
        v->real *= norm1.real;
        v->imag *= norm1.real;
      }
    }

// Household Vector Done

// Intermediate result computing
    {
      complex_t *tmpxq = tmp0;
      complex_t *tmpxr = tmp1;

      SB_CONFIG(mulconj_config, mulconj_size);
      if (i) {
        SB_DMA_READ(sub_q, 0, _N_ * len * 8, 1, P_mulconj_Q);
      }
      SB_DMA_READ(v, 0, 8 * len, _N_, P_mulconj_V);
      SB_CONST(P_mulconj_R, 0, i * len);
      if (i) {
        SB_DMA_READ(r, 0, len * len * 8, 1, P_mulconj_R);
      }

      SB_GARBAGE(P_mulconj_O1, len * i);
      for (y = 0; y < i; ++y) {
        SB_CONST(P_mulconj_reset, 0, len - 1);
        SB_CONST(P_mulconj_reset, 1, 1);
        SB_GARBAGE(P_mulconj_O0, len - 1);
        SB_DMA_WRITE(P_mulconj_O0, 0, 8, 1, tmpxq++);
      }

      for (x = 0; x < len; ++x) {
        if (!i) {
          SB_CONST(P_mulconj_Q, 0, x);
          SB_CONST(P_mulconj_Q, 1065353216, 1);
          SB_CONST(P_mulconj_Q, 0, _N_ - 1 - x);
          SB_DMA_READ(a + x, 8 * _N_, 8, _N_, P_mulconj_R);
        }
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
      complex_t *tmpxq = tmp0, *nxt = sub_q;

      SB_CONFIG(finalize_config, finalize_size);
      if (i) {
        SB_DMA_READ(sub_q, 8, 8, _N_ * len, P_finalize_Q);
      }
      SB_DMA_READ(v, 0, 8 * len, i, P_finalize_VQ);

      SB_CONST(P_finalize_TMPR, 0, i * len);
      SB_CONST(P_finalize_VR, 0, i * len);
      SB_CONST(P_finalize_R, 0, i * len);
      SB_GARBAGE(P_finalize_O1, i * len);

      for (y = 0; y < i; ++y) {
        SB_CONST(P_finalize_TMPQ, *((uint64_t*)tmpxq), len);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, 1, Q + y * _N_ + i);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, len - 1, nxt);
        tmpxq++;
        nxt += len - 1;
      }


      {
        SB_DMA_READ(tmp1, 0, 8 * len, 1, P_finalize_TMPR);
        if (i) {
          SB_DMA_READ(r, len * 8, 8, len, P_finalize_R);
        } else {
          SB_DMA_READ(a, 8, 8, _N_, P_finalize_R);
        }
        SB_CONST(P_finalize_VR, *((uint64_t*)v), len);

        SB_CONST(P_finalize_TMPQ, *((uint64_t*)tmpxq), len);
        SB_DMA_READ(v, 8, 8, len, P_finalize_VQ);


        SB_DMA_WRITE(P_finalize_O1, 8, 8, len, R + i * _N_ + i);
        SB_DMA_WRITE(P_finalize_O0, 0, 8, 1, Q + i * _N_ + i);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, len - 1, nxt);

        ++tmpxq;
        nxt += len - 1;

      }

 

      SB_DMA_READ(v, 0, 8 * len, len - 1, P_finalize_VQ);

      SB_CONST(P_finalize_TMPR, 0, len - 1);
      SB_CONST(P_finalize_VR, 0, len - 1);
      SB_CONST(P_finalize_R, 0, len - 1);
      SB_GARBAGE(P_finalize_O1, len - 1);

      if (i) {
        SB_DMA_READ(r + len + 1, 8 * len, 8 * (len - 1), len - 1, P_finalize_R);
      }
      SB_DMA_READ(v + 1, 0, 8 * (len - 1), len - 1, P_finalize_VR);
      SB_DMA_WRITE(P_finalize_O1, 8, 8, (len - 1) * (len - 1), r);

      if (!i) {
        SB_CONST(P_finalize_Q, 1065353216, 1);
        SB_CONST(P_finalize_Q, 0, _N_ - 1);
      }

      for (y = i + 1; y < _N_; ++y) {
        if (!i) {
          SB_CONST(P_finalize_Q, 0, y);
          SB_CONST(P_finalize_Q, 1065353216, 1);
          SB_CONST(P_finalize_Q, 0, _N_ - 1 - y);
          SB_DMA_READ(a + _N_ + y, 8 * _N_, 8, _N_ - 1, P_finalize_R);
        }
        SB_CONST(P_finalize_TMPQ, *((uint64_t*)tmpxq), len);
        SB_CONST(P_finalize_TMPR, *((uint64_t*)tmp1 + y - i), len - 1);
        SB_DMA_WRITE(P_finalize_O0, 0, 8, 1, Q + y * _N_ + i);
        SB_DMA_WRITE(P_finalize_O0, 8, 8, len - 1, nxt);
        ++tmpxq;
        nxt += len - 1;
      }

      SB_WAIT_ALL();
    }

  }
}
#undef h

