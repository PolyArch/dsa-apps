/**
 * gemm.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sim_timing.h>

#define DATA_TYPE double

DATA_TYPE A[NI * NK], B[NK * NJ], C[NI * NJ];

/* Array initialization. */
static
void init_array(DATA_TYPE *alpha, DATA_TYPE *beta,
                DATA_TYPE C[NI * NJ], DATA_TYPE A[NI * NK], DATA_TYPE B[NK * NJ]) {
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (int i = 0; i < NI; i++)
    for (int j = 0; j < NJ; j++)
      C[i * NI + j] = ((DATA_TYPE) i*j) / NI;
  for (int i = 0; i < NI; i++)
    for (int j = 0; j < NK; j++)
      A[i * NK + j] = ((DATA_TYPE) i*j) / NJ;
  for (int i = 0; i < NK; i++)
    for (int j = 0; j < NJ; j++)
      B[i * NJ + j] = ((DATA_TYPE) i*j) / NK;
}

#ifndef U
#define U 4
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(DATA_TYPE alpha[1], DATA_TYPE beta[1],
		 DATA_TYPE C[NI * NJ], DATA_TYPE A[NI * NK], DATA_TYPE B[NK * NJ]) {

  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < NI; ++i) {
      #pragma ss dfg dedicated unroll(U)
      for (int j = 0; j < NJ; ++j)
        C[i * NJ + j] *= beta[0];
    }
  }

  #pragma ss config
  {
    /* C := alpha*A*B + beta*C */
    for (int i = 0; i < NI; i++) {
      #pragma ss stream
      for (int k = 0; k < NK; ++k) {
        #pragma ss dfg dedicated unroll(U)
        for (int j = 0; j < NJ; j++)
          C[i * NJ + j] += alpha[0] * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

}


int main(int argc, char** argv) {
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  double alpha[1], beta[1];

  /* Initialize array(s). */
  init_array(alpha, beta, C, A, B);

  kernel_gemm(alpha, beta, C, A, B);

  /* Start timer. */
  begin_roi();
  /* Run kernel. */
  kernel_gemm(alpha, beta, C, A, B);
  end_roi();
  sb_stats();


  return 0;
}
