/**
 * gesummv.c: This file is part of the PolyBench/C 3.2 test suite.
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

/* Array initialization. */
void init_array(DATA_TYPE *alpha, DATA_TYPE *beta,
		DATA_TYPE A[N * N], DATA_TYPE B[N * N], DATA_TYPE x[N])
{
  int i, j, n = N;

  *alpha = 43532;
  *beta = 12313;
  for (i = 0; i < n; i++)
    {
      x[i] = ((DATA_TYPE) i) / n;
      for (j = 0; j < n; j++) {
	A[i * N +j] = ((DATA_TYPE) i*j) / n;
	B[i * N +j] = ((DATA_TYPE) i*j) / n;
      }
    }
}

#ifndef U
#define U 4
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gesummv(DATA_TYPE alpha, DATA_TYPE beta,
		    DATA_TYPE A[N * N], DATA_TYPE B[N * N],
		    DATA_TYPE tmp[N], DATA_TYPE x[N], DATA_TYPE y[N])
{
  
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; i++) {
      double acc0, acc1;
      #pragma ss dfg dedicated unroll(U)
      for (int j = 0; j < N; j++) {
        acc0 += A[i * N + j] * x[j];
        acc1 += B[i * N + j] * x[j];
      }
      y[i] = alpha * acc0 + beta * acc1;
    }
  }

}

DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE A[N * N];
DATA_TYPE B[N * N];
DATA_TYPE tmp[N];
DATA_TYPE x[N];
DATA_TYPE y[N];

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */


  /* Initialize array(s). */
  init_array(&alpha, &beta, A, B, x);

  kernel_gesummv(alpha, beta, A, B, tmp, x, y);
  begin_roi();
  kernel_gesummv(alpha, beta, A, B, tmp, x, y);
  end_roi();
  sb_stats();

  return 0;
}
