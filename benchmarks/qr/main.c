#include "qr.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "../../common/include/sim_timing.h"

DTYPE a[N * N], q[N * N], r[N * N], aa[N * N];
int i, j, k;

int main(int argc, char **argv) {
  for (i = 0; i < N * N; ++i) {
    //a[i] = i + 1;
    a[i] = rand() % N + 1;
  }

  begin_roi();
  qr(a, q, r);
  end_roi();

#ifdef DEBUG
  puts("a:");
  for (i = 0; i < N; ++i) { for (j = 0; j < N; ++j) printf("%f ", a[i * N + j]); puts(""); }

  puts("q:");
  for (i = 0; i < N; ++i) { for (j = 0; j < N; ++j) printf("%f ", q[i * N + j]); puts(""); }

  puts("r:");
  for (i = 0; i < N; ++i) { for (j = 0; j < N; ++j) printf("%f ", r[i * N + j]); puts(""); }
#endif

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      for (k = 0; k < N; ++k) {
        aa[i * N + j] += q[i * N + k] * r[k * N + j];
      }
      if (fabs(a[i * N + j] - aa[i * N + j]) > eps) {
        printf("%f %f\n", a[i * N + j], aa[i * N + j]);
        assert(false);
      }
    }
  }

#ifdef DEBUG
  puts("qr:");
  for (i = 0; i < N; ++i) { for (j = 0; j < N; ++j) printf("%f ", aa[i * N + j]); puts(""); }
#endif

  puts("OK!\n");
  return 0;
}
