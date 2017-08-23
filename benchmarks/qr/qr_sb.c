#include "qr.h"
#include "dot.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"

typedef union {
  float c[2];
  double a;
  int64_t b;
} ri_t;
ri_t zero = {0., 0.};
bool active = false;

//float sb_dot(float *a, float *b, int n) {
void sb_dot(int src, float *b, int n, float _cons, float _coef, float *target) {
  if (!n) {
    *target =  _cons;
    //target[1] = 0;
  } else {
    int total = ((n - 1) >> 3) + 1;

    ri_t coef = {_coef, _coef};
    ri_t cons = {_cons, 0};

    //SB_DMA_READ(a, 8, 8, n / 2, P_dot_A);
    SB_SCRATCH_READ(src, sizeof(float) * n, P_dot_A);
    SB_CONST(P_dot_A, 0, (total << 3) - n >> 1);

    SB_DMA_READ(b, 8, 8, n / 2, P_dot_B);
    SB_CONST(P_dot_B, 0, (total << 3) - n >> 1);

    SB_CONST(P_dot_reset, 0, total - 1);
    SB_CONST(P_dot_reset, 1, 1);

    SB_CONST(P_dot_coef, coef.b, total);

    SB_CONST(P_dot_cons, zero.b, total - 1);
    SB_CONST(P_dot_cons, cons.b, 1);

    SB_GARBAGE(P_dot_R, total - 1);
    SB_DMA_WRITE(P_dot_R, 8, 8, 1, target);
 
    //SB_WAIT_ALL();
  }
}

float dot_prod(float *a, float *b, int n) {
  float res = 0.;
  while (n--) {
    res += *a++ * *b++;
  }
  return res;
}

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(DTYPE *a, DTYPE *q, DTYPE *r) {
  SB_CONFIG(dot_config, dot_size);
  int i, j, k, x, y;
  DTYPE *tmp = (DTYPE *) malloc(N * N * 2 * sizeof(DTYPE));
  DTYPE *v = (DTYPE *) malloc((N + 1) * sizeof(DTYPE)),
        *vv= (DTYPE *) malloc((N + 1) * sizeof(DTYPE));

  for (i = 0; i < N; ++i) {
    q[i * (N + 1)] = 1;
  }
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      r[j * N + i] = a[i * N + j];
    }
  }

  for (i = 0; i < N; ++i) {
    //printf("%d\n", i);
    float dot = 0.;
    int n = N - i;
    int nn = n - 1;
    int _n = n + (n & 1);
    int _nn = nn + (nn & 1);

    {
      DTYPE *rp = r + i * (N + 1), *vp = v;
      for (j = i; j < N; ++j) {
        *vp = *rp;
        dot += *vp * *vp;
        ++rp;
        ++vp;
      }
    }

    *v += (r[i * (N + 1)] < -eps ? -1 : 1) * sqrt(dot);

    {
      dot = 0.;
      float *vp = v, *vvp = vv;
      for (j = 0; j < n; ++j) {
        dot += *vp * *vp;
        ++vp;
      }
      dot = sqrt(dot);
      vp = v;
      for (j = 0; j < n; ++j) {
        *vp /= dot;
        if (j) {
          *vvp++ = *vp;
        }
        //printf("%f ", *vp);
        ++vp;
      }
      //puts("");
      *vp = 0;
      *vvp= 0;
      SB_DMA_SCRATCH_LOAD(v, 8, 8, _n >> 1, 0);
      SB_WAIT_SCR_WR();
      SB_DMA_SCRATCH_LOAD(vv, 8, 8, _nn >> 1, _n * sizeof(float));
      SB_WAIT_SCR_WR();
    }

    {
      //begin_roi();
      DTYPE *vk, *vx, *qk, *qx, *res;
      for (y = 0; y < N; ++y) {
        res = tmp + (y * N + i << 1);
        int delta = (y * N + i) & 1;
        for (x = i, vx = v, qx = q + y * N + i; x < N; ++x) {
          //Sequential pattern:
          float coef = -2 * (*vx++);
          sb_dot(
              delta ? _n * sizeof(float) : 0,
              q + y * N + i + delta,
              delta ? _nn :_n,
              (*qx++) + (delta ? (*v * q[y * N + i] * coef) : 0),
              coef,
              res
          );
          res += 2;
          /*
          Origin Q=Q'H:
            for (k = i; k < N; ++k) {
              tmp[y * N + x] += q[y * N + k] * ((k == x) - v[k] * v[x] * 2);
            }
          */
        }
      }
      SB_WAIT_ALL();
      //end_roi();
    }

    {
      for (y = 0; y < N; ++y) {
        float *sum = tmp + (y * N + i << 1);
        for (x = i; x < N; ++x) {
          q[y * N + x] = sum[0];
          sum += 2;
        }
      }
    }

    {
      //begin_roi();
      DTYPE *vy = v, *vk, *rk, *head_rr;
      for (y = i; y < N; ++y) {
        float *res = tmp + (y * N + i << 1);
        for (x = i; x < N; ++x) {
          int delta = (i + x * N) & 1;
          float coef = -2 * *vy;
          sb_dot(
              delta ? _n * sizeof(float) : 0,
              r + i + x * N + delta,
              delta ? _nn :_n,
              (delta ? *v * r[i + x * N] * coef : 0) + r[y + x * N],
              coef,
              res
          );
          res += 2;
          /*
          Origin R=HR':
            for (k = i; k < N; ++k)
              tmp[y * N + x] += h(y, k) * r[k * N + x];
          */
        }
        ++vy;
      }
      SB_WAIT_ALL();
      //end_roi();
    }

    {
      float *sum, *vy = v;
      for (y = i; y < N; ++y) {
        sum = tmp + (y * N + i << 1);
        for (x = i; x < N; ++x) {
          r[x * N + y] = sum[0];
          sum += 2;
        }
        ++vy;
      }
    }

#ifdef DEBUG
    for (y = i; y < N; ++y)
      for (x = i; x < N; ++x)
        for (k = i; k < N; ++k)
          tmp[y * N + x] += h(y, k) * h(k, x);
    puts("h:");
    for (y = 0; y < N; ++y) { for (x = 0; x < N; ++x) printf("%f ", h(y, x)); puts(""); }
    puts("r:");
    for (y = 0; y < N; ++y) { for (x = 0; x < N; ++x) printf("%f ", r[y * N + x]); puts(""); }
    puts("q:");
    for (y = 0; y < N; ++y) { for (x = 0; x < N; ++x) printf("%f ", q[y * N + x]); puts(""); }
    puts("hh:");
    for (y = i; y < N; ++y) { for (x = i; x < N; ++x) printf("%f ", tmp[y * N + x]); puts(""); }
    puts("");
#endif

  }
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      tmp[i * N + j] = r[j * N + i];
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j) {
      r[i * N + j] = tmp[i * N + j];
    }
  //free(v);
  //free(vv);
}
#undef h

