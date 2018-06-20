#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include "../../common/include/sim_timing.h"

#ifndef FxPnt
#define FxPnt 8
#endif

#define INT16MAX ((1<<16)-1)

#define fx_to_flt(x) ((float)(x) / (1<<FxPnt))
#define flt_to_fx(x) ((int)((x) * (1<<FxPnt)))

#ifdef DEBUG
#define fx_mul(a, b) flt_to_fx((fx_to_flt(a) * fx_to_flt(b)))
#else
#define fx_mul(a, b) (a) * (b)
#endif

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

void spmv(int n_a, const Sparse *a, int n_b, const Sparse *b, int16_t *c) {
  //printf("%d %d\n", (int)a[0].delta, (int)a[0].val);
  for (int i = 0, ja = 1; i < N; ++i) {
    for (int jb = 1, idx_a = a[ja - 1].delta + a[ja].delta, idx_b = b[0].delta + b[1].delta;
          a[ja].delta != INT16MAX && b[jb].delta != INT16MAX; ) {
      if (idx_a - a[ja].delta == idx_b) {
        c[i] += fx_mul(a[ja - 1].val, b[jb].val);
      } else if (idx_b - b[jb].delta == idx_a) {
        c[i] += fx_mul(b[jb - 1].val, a[ja].val);
      } else {
        if (idx_a == idx_b) {
          c[i] += fx_mul(b[jb].val, a[ja].val);
        }
        if (idx_a - a[ja].delta == idx_b - b[jb].delta) {
          c[i] += fx_mul(a[ja - 1].val, b[jb - 1].val);
        }
      }
      if (idx_a < idx_b) {
        idx_a += a[++ja].delta;
        idx_a += a[++ja].delta;
      } else if (idx_b < idx_a) {
        idx_b += b[++jb].delta;
        idx_b += b[++jb].delta;
      } else {
        idx_a += a[++ja].delta;
        idx_a += a[++ja].delta;
        idx_b += b[++jb].delta;
        idx_b += b[++jb].delta;
      }
    }
    //printf("%d %d %d %d\n", ja, (int) a[ja].delta, jb, (int) b[jb].delta);
    while(a[ja].delta != INT16MAX)
      ja += 2;
#ifdef DEBUG
    //printf("row %d done\n", i);
    assert (a[ja - 1].delta == INT16MAX);
    if (i != N - 1) {
      assert(a[ja + 1].delta != INT16MAX);
    }
#endif
    ja += 2;
  }
}

#else
// Write softbrain version later
#endif

Sparse a[(int)(N * M * S0 * 1.5)], b[M + 10];
int16_t c[N];
int total_a = 0, total_b = 0;

void input_data() {
  FILE* input = fopen("input.data", "r");
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
}

void check_correctness() {
#ifdef DEBUG
  FILE *output = fopen("output.data", "r");
  for (int i = 0; i < N; ++i) {
    float _c = fx_to_flt(c[i]);
    float ref;
    assert(fscanf(output, "%f", &ref) == 1);
    if(fabs(_c - ref) > 1) {
      printf("expected %f but %f @%d\n", ref, _c, i);
      //assert(false);
    }
  }
#else
  printf("Non-debug mode, correctness check skipped!\n");
#endif
}

int main() {
  input_data();

#ifndef DEBUG
  spmv(total_a, a, total_b, b, c);
#endif
  begin_roi();
  spmv(total_a, a, total_b, b, c);
  end_roi();

  check_correctness();

  return 0;
}
