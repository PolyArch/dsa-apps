#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include "../../common/include/sim_timing.h"
#include "../../common/include/sb_insts.h"

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
 
#ifdef DEBUG
int profile_wei = 0;
int profile_act = 0;
#endif

void spmv(int n_a, const Sparse *a, Sparse **a_head, int n_b, const Sparse *b, int16_t *c) {
  //printf("%d %d\n", (int)a[0].delta, (int)a[0].val);
  for (int i = 0, ja = 1; i < N; ++i) {
#ifndef DEBUG
    ja = (a_head[i] - a) + 1;
#endif
    int jb, idx_a, idx_b;
    for (jb = 1, idx_a = a[ja - 1].delta + a[ja].delta, idx_b = b[0].delta + b[1].delta;
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
#ifdef DEBUG
    while(a[ja].delta != INT16MAX) {
      ja += 2;
      profile_wei++;
    }
    profile_act += (n_b - jb) / 2;
    //printf("row %d done\n", i);
    assert (a[ja - 1].delta == INT16MAX);
    if (i != N - 1) {
      assert(a[ja + 1].delta != INT16MAX);
    }
    ja += 2;
#endif
  }
}

#else

#include "eie.dfg.h"
#include "../../common/include/sb_insts.h"

void spmv(int n_a, const Sparse *a, Sparse **a_head, int n_b, const Sparse *b, int16_t *c) {
  SB_CONFIG(eie_config, eie_size);
  SB_DMA_WRITE(P_eie_O, 0, N * 2, 1, c);
  SB_DMA_READ(b, 0, 4 * n_b, N / 4, P_eie_V);
  //SB_DMA_WRITE(P_eie_O, 8, 8, N / 4, c);
  //printf("%d\n", n_b);
  for (int i = 0; i < N; i += 4) {
    //printf("%d %d %d %d %d\n", i, (a_head[i + 1] - a_head[i + 0]), (a_head[i + 2] - a_head[i + 1]),
        //(a_head[i + 3] - a_head[i + 2]), (a_head[i + 4] - a_head[i + 3]));
    SB_DMA_READ(a_head[i + 0], 0, 4 * (a_head[i + 1] - a_head[i + 0]), 1, P_eie_R0);
    SB_DMA_READ(a_head[i + 1], 0, 4 * (a_head[i + 2] - a_head[i + 1]), 1, P_eie_R1);
    SB_DMA_READ(a_head[i + 2], 0, 4 * (a_head[i + 3] - a_head[i + 2]), 1, P_eie_R2);
    SB_DMA_READ(a_head[i + 3], 0, 4 * (a_head[i + 4] - a_head[i + 3]), 1, P_eie_R3);
  }
  SB_WAIT_ALL();
}

#endif

Sparse a[(int)(N * M * S0 * 1.5)];
Sparse *a_head[N + 10];
Sparse b[M + 10];
int16_t c[N * 4];
int total_a = 0, total_b = 0;

void input_data(int &total_a, Sparse *a, Sparse **a_head, int &total_b, Sparse *b) {
  FILE* input = fopen("input.data", "r");
#define total total_a
  for (int i = 0; i < N; ++i) {
    a_head[i] = a + total_a;
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
  a_head[N] = a + total;
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
  printf("Empty runs of Weight: %d\n", profile_wei);
  printf("Empty runs of Activation: %d\n", profile_act);
#else
  printf("Non-debug mode, correctness check skipped!\n");
#endif
}

int main() {
  input_data(total_a, a, a_head, total_b, b);

#ifndef DEBUG
  spmv(total_a, a, a_head, total_b, b, c);
#endif
  begin_roi();
  spmv(total_a, a, a_head, total_b, b, c);
  end_roi();
  sb_stats();

  check_correctness();

  return 0;
}
