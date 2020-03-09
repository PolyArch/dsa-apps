/**
 * 3mm.c: This file is part of the PolyBench/C 3.2 test suite.
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


DATA_TYPE A[NI * NJ];
DATA_TYPE B[NJ * NK];
DATA_TYPE C[NK * NL];
DATA_TYPE D[NL * NM];
DATA_TYPE E[NI * NK];
DATA_TYPE F[NK * NM];
DATA_TYPE G[NI * NM];

/* Array initialization. */
void init_array(DATA_TYPE A[NI * NJ],
		DATA_TYPE B[NJ * NK],
		DATA_TYPE C[NK * NL],
		DATA_TYPE D[NL * NM])
{
  int i, j;
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      A[i * nj + j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nk; j++)
      B[i * nk + j] = ((DATA_TYPE) i*(j+1)) / nj;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nl; j++)
      C[i * nl + j] = ((DATA_TYPE) i*(j+3)) / nk;
  for (i = 0; i < nl; i++)
    for (j = 0; j < nm; j++)
      D[i * nm + j] = ((DATA_TYPE) i*(j+2)) / nl;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_3mm(DATA_TYPE E[NI * NK],
		DATA_TYPE A[NI * NJ],
		DATA_TYPE B[NJ * NK],
		DATA_TYPE F[NK * NM],
		DATA_TYPE C[NK * NL],
		DATA_TYPE D[NL * NM],
		DATA_TYPE G[NI * NM])
{



  /* F := C*D */
  #pragma ss config
  {
    for (int i = 0; i < NK; i++) {
      #pragma ss stream
      for (int k = 0; k < NL; ++k) {
        #pragma ss dfg dedicated unroll(U)
        for (int j = 0; j < NM; j++)
          F[i * NM + j] += C[i* NL + k] * D[k * NM + j];
      }
    }
  }

  {
    for (int i = 0; i < NI; i++) {
      double E[NK];
      /* E := A*B */
      #pragma ss config
      {
        #pragma ss stream
        for (int k = 0; k < NJ; ++k) {
          #pragma ss dfg dedicated unroll(U)
          for (int j = 0; j < NK; j++)
            E[j] += A[i * NJ + k] * B[k * NK + j];
        }
      }
      /* G := E*F */
      #pragma ss config
      {
        #pragma ss stream
        for (int k = 0; k < NK; ++k)
          #pragma ss dfg dedicated unroll(U)
          for (int j = 0; j < NM; j++)
            G[i * NM + j] += E[k] * F[k * NM + j];
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
  int nm = NM;

  /* Initialize array(s). */
  init_array (A, B, C, D);

  /* Start timer. */

  kernel_3mm(E, A, B, F, C, D, G);
  /* Run kernel. */
  begin_roi();
  kernel_3mm(E, A, B, F, C, D, G);
  end_roi();
  sb_stats();

  return 0;
}
