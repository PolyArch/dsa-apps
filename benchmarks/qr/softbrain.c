#include "qr.h"
#include "compute.h"
#include "sb_insts.h"
#include "sim_timing.h"

typedef union {
  float c[2];
  double a;
  int64_t b;
} ri_t;
ri_t zero = {0., 0.};
bool active = false;

//float sb_dot(float *a, float *b, int n) {
void sb_dot(int src, float *b, int n, float *target) {
  end_roi();
  {
    //Do some strange memory access to get rid of the bug of simulator!
    uint64_t x = src + (uint64_t) b + n + (uint64_t) target;
    --x;
  }
  begin_roi();
  if (!n) {
    *target = 0;
    //target[1] = 0;
  } else {
    int total = ((n - 1) >> 3) + 1;

    //SB_DMA_READ(a, 8, 8, n / 2, P_compute_A);
    SB_SCRATCH_READ(src, sizeof(float) * n, P_compute_A);
    SB_CONST(P_compute_A, 0, (total << 3) - n >> 1);

    SB_DMA_READ(b, 8, 8, n / 2, P_compute_B);
    SB_CONST(P_compute_B, 0, (total << 3) - n >> 1);

    SB_CONST(P_compute_reset, 0, total - 1);
    SB_CONST(P_compute_reset, 1, 1);

    SB_GARBAGE(P_compute_R, total - 1);
    SB_DMA_WRITE(P_compute_R, 8, 8, 1, target);
 
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
  SB_CONFIG(compute_config, compute_size);
  int i, j, k, x, y;
  DTYPE *tmp = (DTYPE *) malloc(N * N * 2 * sizeof(DTYPE));
  DTYPE *v = (DTYPE *) malloc((N + 1) * sizeof(DTYPE)),
        *vv= (DTYPE *) malloc((N + 1) * sizeof(DTYPE)),
        *fly= (DTYPE *) malloc((N + 1) * sizeof(DTYPE));

  for (i = 0; i < N; ++i) {
    q[i * (N + 1)] = 1;
  }
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      r[j * N + i] = a[i * N + j];
    }
  }

  for (i = 0; i < N; ++i) {
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
        ++vp;
      }
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
        fly[y] = q[y * N + i];
        int delta = (y * N + i) & 1;
        for (x = i, vx = v, qx = q + y * N + i; x < N; ++x) {
          //Sequential pattern:
          sb_dot(
              delta ? _n * sizeof(float) : 0,
              q + y * N + i + delta,
              delta ? _nn :_n,
              res
          );

          res += 2;

          /*
          Origin_QQH:
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
        float *vx = v;
        int delta = y * N + i & 1;
        for (x = i; x < N; ++x) {
          q[y * N + x] += (sum[0] + (delta ? fly[y] * *v : 0)) * -2 * *vx++;
          sum += 2;
        }
      }
    }

    {
      //begin_roi();
      DTYPE *vy = v, *vk, *rk, *head_rr;
      for (y = i; y < N; ++y) {
        float *res = tmp + (y * N + i << 1);
        fly[y] = r[i + y * N];
        for (x = i; x < N; ++x) {
          int delta = (i + x * N) & 1;
          sb_dot(
              delta ? _n * sizeof(float) : 0,
              r + i + x * N + delta,
              delta ? _nn :_n,
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
          int delta = i + x * N & 1;
          r[x * N + y] += (sum[0] + (delta ? *v * fly[x] : 0)) * -2 * *vy;
          sum += 2;
        }
        ++vy;
      }
    }

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

