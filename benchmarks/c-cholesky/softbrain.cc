#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "sb-2-wide/compute.h"
#include "sb_insts.h"

using std::complex;

struct complex_t {
  float real, imag;
};

void cholesky(complex<float> *a, complex<float> *L) {
  SB_CONFIG(compute_config, compute_size);

  complex_t *A = new complex_t[N * N];
  for (int i = 0; i < N * N; ++i) {
    A[i].real = a[i].real();
    A[i].imag = a[i].imag();
  }
  for (int i = 0; i < N; ++i) {
    complex_t div = {1, 0};
    {
      //complex_t div = std::sqrt(A[i * (N + 1)]);
      complex_t v = A[i * (N + 1)];
      complex_t pre = {-1, 0};
      complex_t tmp;
      float delta = 1.;
      //std::cout << v.real << ", " << v.imag << "\n";
      while (delta > eps) {
        pre = div;
        // div = (div + v / div) / 2
        // 0. tmp = v / div
        float norm = div.real * div.real + div.imag * div.imag;
        tmp.real = (v.real * div.real + v.imag * div.imag) / norm;
        tmp.imag = (v.imag * div.real - v.real * div.imag) / norm;

        // 1. div += tmp
        div.real += tmp.real;
        div.imag += tmp.imag;
        // 2. div /= 2
        div.real /= 2;
        div.imag /= 2;
        //update delta
        tmp.real = fabs(pre.real - div.real);
        tmp.imag = fabs(pre.imag - div.imag);
        delta = tmp.real * tmp.real + tmp.imag * tmp.imag;
      }
    }
    complex_t *b = A + i * (N + 1);
    L[i * (N + 1)] = std::complex<float>(div.real, div.imag);
    {
      complex_t *bp = b + 1;
      float norm = 1 / ((div.real * div.real) + (div.imag * div.imag));
      for (int j = i + 1; j < N; ++j) {
        //L[j * N + i] = b[j - i] / div;
        L[j * N + i] = complex<float>(
            (bp->real * div.real + bp->imag * div.imag) * norm,
            (bp->imag * div.real - bp->real * div.imag) * norm
        );
        ++bp;
      }
    }
    {
      complex_t *bj = b + 1, *bk, v = A[i * (N + 1)];
      float norm = 1 / (v.real * v.real + v.imag * v.imag);
      union {
        float f[2];
        uint64_t v;
      } ri_norm = {-norm, -norm}, ri_v = {v.real, v.imag};
      for (int j = i + 1; j < N; ++j) {
        int times = N - i - 1;
        bk = b + 1;
        uint64_t ri_bj = *((uint64_t *) bj);

        SB_CONST(P_compute__A, ri_bj, times);
        SB_DMA_READ(bk, 8, 8, times, P_compute__B);
        SB_CONST(P_compute__V, ri_v.v, times);
        SB_CONST(P_compute_NORM, ri_norm.v, times);
        SB_DMA_READ(A + j * N + i + 1, 8, 8, times, P_compute_Z);
        SB_DMA_WRITE(P_compute_O, 8, 8, times, A + j * N + i + 1);

/*
        for (int k = i + 1; k < N; ++k) {
          //A[j * N + k] -= std::conj(b[j - i]) * b[k - i] / A[i * (N + 1)];
          
          //tmp = bj* x bk
          complex_t tmp;
          tmp.real = bj->real * bk->real + bj->imag * bk->imag;
          tmp.imag = bj->real * bk->imag - bj->imag * bk->real;

          //tmp /= v
          tmp = (complex_t) {
            (tmp.real * v.real + tmp.imag * v.imag) * norm,
            (tmp.imag * v.real - tmp.real * v.imag) * norm
          };

          //A[j * N + k].real -= tmp.real;
          //A[j * N + k].imag -= tmp.imag;

          ++bk;
        }
*/
        ++bj;
      }
      // Aggressive wait
      if (N - i < 4)
        SB_WAIT_ALL();
    }
  }
  delete []A;
}

