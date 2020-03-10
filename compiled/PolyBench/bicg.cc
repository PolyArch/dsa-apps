/**
 * bicg.c: This file is part of the PolyBench/C 3.2 test suite.
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

#ifndef U
#define U 2
#endif

#define DATA_TYPE double

void init_array (DATA_TYPE A[NI * NJ], DATA_TYPE r[NI], DATA_TYPE p[NJ]) {
  int i, j;
  double pi = atan(1) * 4;

  for (int i = 0; i < NJ; i++)
    p[i] = i * pi;
  for (int i = 0; i < NI; i++) {
    r[i] = i * pi;
    for (int j = 0; j < NJ; j++)
      A[i * NJ + j] = ((DATA_TYPE) i*(j+1))/NI;
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_bicg(DATA_TYPE A[NI * NJ], DATA_TYPE s[NJ], DATA_TYPE q[NI], DATA_TYPE p[NJ],
                 DATA_TYPE r[NI]) {

  #pragma ss config
  {
    #pragma ss stream
    for (int i = 0; i < NI; i++) {
      double acc = 0;
      #pragma ss dfg dedicated unroll(U)
      for (int j = 0; j < NJ; j++) {
        s[j] = s[j] + r[i] * A[i * NJ + j];
        acc += A[i * NJ + j] * p[j];
      }
      q[i] = acc;
    }
  }

}

DATA_TYPE A[NI * NJ], s[NJ], q[NI], p[NJ], r[NI];

int main(int argc, char** argv)
{

  /* Initialize array(s). */
  init_array (A, r, p);

  kernel_bicg(A, s, q, p, r);
  begin_roi();
  /* Run kernel. */
  kernel_bicg(A, s, q, p, r);
  end_roi();
  sb_stats();

  return 0;
}
