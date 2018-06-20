#include <cstdio>
#include <cstdint>
#include <cassert>
#include "../../common/include/sim_timing.h"

#ifndef FxPnt
#define FxPnt 10
#endif

#define INT16MAX ((1<<16)-1)

#define fx_to_flt(x) ((float)(x) / (1<<FxPnt))

struct Sparse {
  uint16_t delta;
  int16_t val;
};

void dump(int n, const Sparse *a) {
  for (int i = 0; i < n; ++i) {
    printf("%d %f\n", (int)a[i].delta, fx_to_flt(a[i].val));
  }
  puts("========");
}


#ifdef __x86_64__

void spmv(int n_a, const Sparse *a, int n_b, const Sparse *b, uint16_t *c) {
  dump(n_a, a);
  //printf("%d %d\n", (int)a[0].delta, (int)a[0].val);
  for (int i = 0, ja = 0; i < N; ++i) {
    int jb, idx_a, idx_b;
    for (jb = 0, idx_a = a[ja].delta, idx_b = a[jb].delta;
          a[ja].delta != INT16MAX && b[jb].delta != INT16MAX; ) {
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
    //printf("%d %d %d %d\n", ja, (int) a[ja].delta, jb, (int) b[jb].delta);
    while(a[ja].delta != INT16MAX)
      ++ja;
    assert (a[ja].delta == INT16MAX);
    ++ja;
    assert(a[ja].delta == INT16MAX);
    ++ja;
    if (i == N - 1) {
      //printf("%d %d %d\n", i, ja, n_a);
      assert(ja == n_a);
    } else {
      //printf("%d %d\n", i, ja);
    }
  }
}

#else
// Write softbrain version later
#endif

Sparse a[(int)(N * M * S0 * 1.5)], b[M + 10];
uint16_t c[N];

int main() {
  FILE* input = fopen("input.data", "r");
  int total_a = 0;
#define total total_a
  for (int i = 0; i < N; ++i) {
    int delta;
    float val;
    while (true) {
      assert(fscanf(input, "%d%f", &delta, &val) == 2);
      if (delta == -1) {
        a[total++] = (Sparse) { (uint16_t)INT16MAX, (int16_t)(-1) };
        a[total++] = (Sparse) { (uint16_t)INT16MAX, (int16_t)(-1) };
        //printf("%d\n", total);
        break;
      } else {
        a[total++] = (Sparse) {(uint16_t)delta, (int16_t)(val * (1 << FxPnt))};
        //printf("%d %d\n", (int)a[total-1].delta, (int)a[total-1].val);
      }
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
      b[total++] = (Sparse) { (uint16_t)INT16MAX, (int16_t)(-1) };
      b[total++] = (Sparse) { (uint16_t)INT16MAX, (int16_t)(-1) };
      break;
    } else {
      b[total++] = (Sparse) {(uint16_t)delta, (int16_t)(val * (1 << FxPnt))};
    }
  }
#undef total
  
  dump(total_a, a);
  spmv(total_a, a, total_b, b, c);
  begin_roi();
  spmv(total_a, a, total_b, b, c);
  end_roi();

  return 0;
}
