#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <complex>
#include "cholesky.h"
#include "compute.h"
#include "offload.h"
#include "writeback.h"
#include "sb_insts.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

using std::complex;

void cholesky(complex<float> *a, complex<float> *L) {
  SB_CONTEXT(1);
  SB_CONFIG(compute_config, compute_size);
  SB_CONTEXT(2);
  SB_CONFIG(offload_config, offload_size);
  SB_CONTEXT(4);
  SB_CONFIG(writeback_config, writeback_size);
  complex<float> div;
  {
    SB_CONTEXT(2);
    SB_DMA_READ(a, 0, 8 * N, 1, P_offload_V);
    SB_CONST(P_offload_MUX, 0, 1);
    SB_CONST(P_offload_MUX, 1, N - 1);

    SB_REPEAT_PORT(N * (N - 1) / 2);
    SB_XFER_LEFT(P_offload_INV, P_compute_V, 1);

    SB_DMA_WRITE(P_offload_SQRT, 0, 8, 1, L);

    SB_REPEAT_PORT(N - 1);
    SB_XFER_RIGHT(P_offload_SQRTINV, P_writeback_DIV, 1);
    SB_XFER_RIGHT(P_offload_STREAM, P_writeback_BP, N - 1);

    SB_CONTEXT(1);
    complex<float> *bj = a + 1, *bk, v = *a;
    SB_DMA_READ_STRETCH(a + N + 1, 8 * (N + 1), 8 * (N - 1), -8, N - 1, P_compute_Z);
    SB_DMA_READ_STRETCH(a + 1,               8, 8 * (N - 1), -8, N - 1, P_compute_B);
    for (int j = 1; j < N; ++j)
      SB_CONST(P_compute_A, *((uint64_t*)a + j), N - j);

    SB_CONTEXT(4);
    SB_DMA_WRITE(P_writeback_RES, N * 8, 8, N - 1, L + N);

    /*SB_CONTEXT(1);
    SB_GARBAGE(P_compute_O0, N * (N - 1) / 2);
    SB_GARBAGE(P_compute_O1, N * (N - 1) / 2);
    SB_CONTEXT(7);
    SB_WAIT_ALL();
    return;*/
  }
  SB_WAIT_COMPUTE();

  int addr = (N - 1) * 8;
  for (int i = 1; i < N; ++i) {
    int total = N - i - 1;
    SB_CONTEXT(1);
    /*SB_GARBAGE(P_compute_O0, 1);
    SB_GARBAGE(P_compute_O0, total);
    SB_GARBAGE(P_compute_O1, N * (N - 1) / 2);*/
    SB_GARBAGE(P_compute_O0, 1);
    //complex<float> tmp;
    //SB_RECV(P_compute_O0, tmp);
    //std::cout << tmp << "\n";
    SB_SCR_WRITE(P_compute_O0, total * 8, addr);
    SB_RECURRENCE(P_compute_O0, P_compute_Z, total * (total + 1) / 2);
    //SB_GARBAGE(P_compute_O0, (total + 1) * total / 2);
    SB_XFER_RIGHT(P_compute_O1, P_offload_V, total + 1);
    //SB_GARBAGE(P_compute_O1, total + 1);
    SB_GARBAGE(P_compute_O1, total * (total + 1) / 2);
    //SB_WAIT_SCR_WR();
    //break;


    SB_CONTEXT(2);
    SB_CONST(P_offload_MUX, 0, 1);
    SB_CONST(P_offload_MUX, 1, total);
    //SB_GARBAGE(P_offload_STREAM, total);
    //SB_GARBAGE(P_offload_INV, 1);
    //SB_GARBAGE(P_offload_SQRTINV, 1);
    //SB_GARBAGE(P_offload_SQRT, 1);

    if (total) {
      SB_REPEAT_PORT((1 + total) * total / 2);
      SB_XFER_LEFT(P_offload_INV, P_compute_V, 1);

      SB_REPEAT_PORT(total);
      SB_XFER_RIGHT(P_offload_SQRTINV, P_writeback_DIV, 1);

    } else {
      SB_GARBAGE(P_offload_INV, 1);
      SB_GARBAGE(P_offload_SQRTINV, 1);
    }

    SB_XFER_RIGHT(P_offload_STREAM, P_writeback_BP, total);
    SB_DMA_WRITE(P_offload_SQRT, 0, 8, 1, L + i * (N + 1));
   

    SB_CONTEXT(1);
    SB_WAIT_SCR_WR();

    //SB_SCR_PORT_STREAM_STRETCH(addr, 8, 8 * total, -8, total, P_compute_B);
    SB_GARBAGE(P_compute_O2, 1);

    for (int j = i + 1, cur = addr; j < N; ++j) {
      int len = N - j;
      SB_SCR_PORT_STREAM(cur, 0, 8, len, P_compute_A);
      SB_REPEAT_PORT(len);
      SB_RECURRENCE(P_compute_O2, P_compute_B, 1);
      cur += 8;
    }
    SB_GARBAGE(P_compute_O2, total * (total + 1) / 2);
    addr += total * 8;

    SB_CONTEXT(4);
    //SB_GARBAGE(P_writeback_RES, total);
    
    SB_DMA_WRITE(P_writeback_RES, N * 8, 8, total, L + (i + 1) * N + i);
    //break;

  }
  SB_CONTEXT(7);
  SB_WAIT_ALL();

}

