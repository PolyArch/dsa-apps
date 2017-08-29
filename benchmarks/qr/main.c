#include "qr.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "../../common/include/sim_timing.h"

float a[N * N], q[N * N], r[N * N];
int i, j, k;

int main(int argc, char **argv) {
  FILE *input = fopen("input.data", "r");
  assert(input && "Input open failed!\n");
  for (i = 0; i < N * N; ++i) {
    assert(fscanf(input, "%f", a + i) == 1 && "Input error!");
  }

  begin_roi();
  qr(a, q, r);
  end_roi();

  FILE *ref = fopen("ref.data", "r");
  assert(input && "Ref open failed!\n");
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      float value;
      fscanf(ref, "%f", &value);
      if (fabs(value - q[i * N + j]) > eps) {
        printf("Error @%d, %d\n%f expected but %f got!\n", i, j, value, q[i * N + j]);
        return 1;
      }
    }
  }
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      float value;
      fscanf(ref, "%f", &value);
      if (fabs(value - r[i * N + j]) > eps) {
        printf("Error @%d, %d\n%f expected but %f got!\n", i, j, value, r[i * N + j]);
        return 1;
      }
    }
  }

  puts("OK!\n");
  return 0;
}
