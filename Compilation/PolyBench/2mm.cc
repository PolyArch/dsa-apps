/**
 * 2mm.c: This file is part of the PolyBench/C 3.2 test suite.
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

#ifndef U
#define U 4
#endif

DATA_TYPE A[NI * NK], B[NK * NJ], C[NJ * NL], D[NI * NL];

void init_array(DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE A[NI * NK],
		DATA_TYPE B[NK * NJ],
		DATA_TYPE C[NJ * NL],
		DATA_TYPE D[NI * NL])
{
  int i, j, ni = NI, nj = NJ, nk = NK, nl = NL;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i * nk + j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i * nj + j] = ((DATA_TYPE) i*(j+1)) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i * nl + j] = ((DATA_TYPE) i*(j+3)) / nk;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i * nl + j] = ((DATA_TYPE) i*(j+2)) / nl;
}


void kernel_2mm(DATA_TYPE alpha[], DATA_TYPE beta, DATA_TYPE A[NI * NK], DATA_TYPE B[NK * NJ],
                DATA_TYPE C[NJ * NL], DATA_TYPE D[NI * NL]) {

  /* D := alpha*A*B*C + beta*D */

  {

    for (int i = 0; i < NI; i++) {
    
      //#pragma ss stream
      //#pragma ss dfg
      //for (j = 0; j < nj; ++j)
      //  tmp[j] = 0;

      DATA_TYPE tmp[NJ];

      #pragma ss config
      {
        #pragma ss stream
        for (int k = 0; k < NK; ++k) {
          #pragma ss dfg dedicated unroll(U)
          for (int j = 0; j < NJ; ++j) {
            tmp[j] += A[i * NK + k] * B[k * NJ + j];
          }
        }
      }

      #pragma ss config
      {
        #pragma ss stream
        #pragma ss dfg dedicated unroll(U)
        for (int j = 0; j < NL; j++)
          D[i * NL + j] += D[i * NL + j] * beta;
      }

      #pragma ss config
      {
        #pragma ss stream
        for (int k = 0; k < NJ; ++k) {
          #pragma ss dfg dedicated unroll(U)
          for (int j = 0; j < NL; j++) {
            D[i * NL + j] += *alpha * tmp[k] * C[k * NL + j];
          }
        }
      }

    }
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  double alpha, beta;

  /* Initialize array(s). */
  init_array (&alpha, &beta, A, B, C, D);

  /* Warm cache. */
  kernel_2mm(&alpha, beta, A, B, C, D);

  begin_roi();
  /* Run kernel. */
  kernel_2mm(&alpha, beta, A, B, C, D);
  end_roi();
  sb_stats();

  return 0;
}
