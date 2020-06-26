/**
 * mvt.c: This file is part of the PolyBench/C 3.2 test suite.
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
void init_array(DATA_TYPE x1[N], DATA_TYPE x2[N],
                DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE A[N * N])
{
  int i, j, n = N;

  for (i = 0; i < n; i++)
    {
      x1[i] = ((DATA_TYPE) i) / n;
      x2[i] = ((DATA_TYPE) i + 1) / n;
      y1[i] = ((DATA_TYPE) i + 3) / n;
      y2[i] = ((DATA_TYPE) i + 4) / n;
      for (j = 0; j < n; j++)
	A[i * N + j] = ((DATA_TYPE) i*j) / N;
    }
}

#ifndef U
#define U 2
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_mvt(DATA_TYPE x1[N],
		DATA_TYPE x2[N],
		DATA_TYPE y1[N],
		DATA_TYPE y2[N],
		DATA_TYPE A[N * N])
{
  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < N; i++) {
      double acc = 0.0;
      #pragma ss dfg dedicated unroll(U)
      for (int j = 0; j < N; j++) {
        acc += A[i * N + j] * y1[j];
        x2[j] = x2[j] + A[i * N + j] * y2[i];
      }
      x1[i] = acc;
    }
  }
}


DATA_TYPE A[N * N];
DATA_TYPE x1[N];
DATA_TYPE x2[N];
DATA_TYPE y1[N];
DATA_TYPE y2[N];

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Initialize array(s). */
  init_array (x1, x2, y1, y2, A);

  kernel_mvt(x1, x2, y1, y2, A);
  begin_roi();
  kernel_mvt(x1, x2, y1, y2, A);
  end_roi();
  sb_stats();

  return 0;
}
