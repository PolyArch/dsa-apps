#include <cstdio>
#include <cstdint>
#include <cassert>
#include "../../common/include/sim_timing.h"

#ifndef FxPnt
#define FxPnt 10
#endif

#define INT16MAX ((1<<16)-1)

struct Sparse {
  uint16_t delta;
  uint16_t val;
};


#ifdef __x86_64__

void spmv(Sparse *a, int nb, Sparse *b, uint16_t *c) {
  for (int i = 0, ja = 0; i < N; ++i) {
    for (int jb = 0, idx_a = a[ja].delta, idx_b = a[jb].delta;
          a[ja].delta != INT16MAX && a[jb].delta != INT16MAX; ) {
      if (idx_a < idx_b) {
        idx_a += a[++ja].delta;
      } else if (idx_b < idx_a) {
        idx_b += b[++jb].delta;
      } else {
        c[i] += a[ja].val * b[jb].val;
        idx_a += a[++ja].delta;
        idx_b += b[++jb].delta;
      }
    }
    while(a[ja].delta != INT16MAX)
      ++ja;
    assert (a[ja].delta == INT16MAX);
    ++ja;
    assert(a[ja].delta == INT16MAX);
    ++ja;
  }
}

#else
// Write softbrain version later
#endif

Sparse a[(int)(N * M * S0 * 1.5)], b[M];
uint16_t c[N];

int main() {
  FILE* input = fopen("input.data", "r");
  int total_a = 0;
  int i = 0;
#define total total_a
  while (i < N) {
    int delta;
    float val;
    assert(fscanf(input, "%d%f", &delta, &val) == 2);
    if (delta == -1) {
      ++i;
      *((int64_t*)(a + total)) = -1;
      total += 2;
    } else {
      a[total++] = (Sparse) {(uint16_t)delta, (uint16_t)(val * (1 << FxPnt))};
    }
  }
#undef total

  int total_b = 0;
#define total total_b
  while (true) {
    int delta;
    float val;
    assert(fscanf(input, "%d%f", &delta, &val) == 2);
    if (delta == -1) {
      *((int64_t*)(b + total)) = -1;
      total += 2;
      break;
    } else {
      b[total++] = (Sparse) {(uint16_t)delta, (uint16_t)(val * (1 << FxPnt))};
    }
  }
#undef total
  begin_roi();
  spmv(a, total_b, b, c);
  end_roi();
  //printf("%d %d\n", total_a, total_b);
  return 0;
}
