/**
 * atax.c: This file is part of the PolyBench/C 3.2 test suite.
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

#ifndef U1
#define U1 2
#endif

#ifndef U2
#define U2 4
#endif

DATA_TYPE A[NI * NJ], x[NI], y[NJ], tmp[NJ];

/* Array initialization. */
void init_array(DATA_TYPE A[NI * NJ], DATA_TYPE x[NJ])
{
  int i, j;
  int nx = NI, ny = NJ;

  double pi = atan(1) * 4.0;
  for (i = 0; i < ny; i++)
      x[i] = i * pi;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i * NJ + j] = ((DATA_TYPE) i*(j+1)) / nx;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_atax(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{

  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < NI; ++i) {
      double acc = 0.0;
      #pragma ss dfg dedicated unroll(U1)
      for (int j = 0; j < NJ; ++j) {
        acc += A[i * NJ + j] * x[j];
        y[j] += A[i * NJ + j];
      }
      tmp[i] = acc;
    }

    #pragma ss stream
    #pragma ss dfg dedicated unroll(U2)
    for (int i = 0; i < NJ; ++i) {
      y[i] *= tmp[i];
    }
  }


}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nx = NI;
  int ny = NJ;

  /* Initialize array(s). */
  init_array (A, x);

  /* Start timer. */
  kernel_atax (A, x, y, tmp);
  begin_roi();
  /* Run kernel. */
  kernel_atax (A, x, y, tmp);
  end_roi();
  sb_stats();

  return 0;
}
