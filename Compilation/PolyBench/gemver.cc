/**
 * gemver.c: This file is part of the PolyBench/C 3.2 test suite.
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
static
void init_array (DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE A[N * N],
		 DATA_TYPE u1[N], DATA_TYPE v1[N], DATA_TYPE u2[N], DATA_TYPE v2[N],
		 DATA_TYPE w[N], DATA_TYPE x[N], DATA_TYPE y[N], DATA_TYPE z[N])
{
  int n = N;
  int i, j;

  *alpha = 43532;
  *beta = 12313;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = (i+1)/n/2.0;
      v1[i] = (i+1)/n/4.0;
      v2[i] = (i+1)/n/6.0;
      y[i] = (i+1)/n/8.0;
      z[i] = (i+1)/n/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
	A[i * N + j] = ((DATA_TYPE) i*j) / n;
    }
}

#ifndef U
#define U 4
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gemver(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N * N],
		 DATA_TYPE u1[N], DATA_TYPE v1[N], DATA_TYPE u2[N], DATA_TYPE v2[N],
		 DATA_TYPE w[N], DATA_TYPE x[N], DATA_TYPE y[N], DATA_TYPE z[N])
{

  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; i++) {
      #pragma ss dfg dedicated unroll(U)
      for (int j = 0; j < N; j++)
        A[i * N + j] = A[i * N + j] + u1[i] * v1[j] + u2[i] * v2[j];
    }
  }

  #pragma ss config
  {
    #pragma ss stream
    for (int j = 0; j < N; j++) {
      #pragma ss dfg dedicated unroll(U)
      for (int i = 0; i < N; i++)
        x[i] = x[i] + beta * A[j * N + i] * y[j];
    }
  }

  #pragma ss config
  {
    #pragma ss stream
    #pragma ss dfg dedicated unroll(U)
    for (int i = 0; i < N; i++)
      x[i] = x[i] + z[i];
  }

  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; i++) {
      double acc = 0.;
      #pragma ss dfg dedicated unroll(U)
      for (int j = 0; j < N; j++)
        acc +=  alpha * A[i * N + j] * x[j];
      w[i] += acc;
    }
  }

}

DATA_TYPE A[N * N];
DATA_TYPE u1[N];
DATA_TYPE v1[N];
DATA_TYPE u2[N];
DATA_TYPE v2[N];
DATA_TYPE w[N];
DATA_TYPE x[N];
DATA_TYPE y[N];
DATA_TYPE z[N];

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;

  /* Initialize array(s). */
  init_array(&alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);


  kernel_gemver(alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  begin_roi();
  kernel_gemver(alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  end_roi();
  sb_stats();

  return 0;
}
