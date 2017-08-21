#include "qr.h"
#include "dot.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"

union {
  float a;
  int b;
} ri = {0.};

float sb_dot(float *a, float *b, int n) {
  float res = 0.;
  int i;
  if (n < 4) {
    for (i = 0; i < n; ++i) {
      res += (*a++) * (*b++);
    }
    return res;
  } else if (n >= 44) {
    float sum[22];
    //SB_CONFIG(dot_config, dot_size);
    SB_CONST(P_dot_carry, ri.b, 11);
    SB_DMA_READ(a, 16, 16, n / 4, P_dot_A);
    SB_DMA_READ(b, 16, 16, n / 4, P_dot_B);
    SB_RECURRENCE(P_dot_R, P_dot_carry, n / 4 - 11);
    SB_DMA_WRITE(P_dot_R, 8, 8, 11, sum);
    SB_WAIT_ALL();
    {
      float *head = sum;
      for (i = 0; i < 11; ++i) {
        res += *head;
        head += 2;
      }
      float *head_a = a + (n >> 2 << 2);
      float *head_b = b + (n >> 2 << 2);
      for (i = 0; i < (n & 3); ++i) {
        res += (*head_a++) * (*head_b++);
      }
    }
  } else {
    float sum[2] = {0, 0};
    //SB_CONFIG(dot_config, dot_size);
    SB_CONST(P_dot_carry, ri.b, 1);
    SB_DMA_READ(a, 16, 16, n / 4, P_dot_A);
    SB_DMA_READ(b, 16, 16, n / 4, P_dot_B);
    SB_RECURRENCE(P_dot_R, P_dot_carry, n / 4 - 1);
    SB_DMA_WRITE(P_dot_R, 8, 8, 1, sum);
    SB_WAIT_ALL();
    {
      /*float *head = b;
      for (i = 0; i < 11; ++i) {
        res += *head;
        head += 2;
      }*/
      res = *sum;
      float *head_a = a + (n >> 2 << 2);
      float *head_b = b + (n >> 2 << 2);
      for (i = 0; i < (n & 3); ++i) {
        res += (*head_a++) * (*head_b++);
      }
    }
  }
  return res;
}


#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(DTYPE *a, DTYPE *q, DTYPE *r) {
  SB_CONFIG(dot_config, dot_size);
  int i, j, k, x, y; DTYPE *tmp = (DTYPE *) malloc(N * N * sizeof(DTYPE));
  DTYPE *v = (DTYPE *) malloc(N * sizeof(DTYPE)),
        *vv= (DTYPE *) malloc(N * sizeof(DTYPE));
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

    //begin_roi();
    dot = sb_dot(v, v, N - i);
    //end_roi();

    dot = sqrt(dot);
    for (j = 0; j < N - i; ++j) {
      v[j] /= dot;
      if (j) {
        vv[j - 1] = v[j];
      }
    }

    for (y = 0; y < N; ++y) for (x = i; x < N; ++x) tmp[y * N + x] = 0;

    {
      //begin_roi();
      DTYPE *vk, *vx, *qk, *qx;
      for (y = 0; y < N; ++y) {
        for (x = i, vx = v, qx = q + y * N + i; x < N; ++x) {
          //Sequential pattern:
          if ((y * N + i) & 1) {
            tmp[y * N + x] =
              (q[y * N + i] * (*v) + sb_dot(q + y * N + i + 1, vv, N - i - 1)) * -2 * (*vx++) + (*qx++);
          } else {
            tmp[y * N + x] = sb_dot(q + y * N + i, v, N - i) * -2 * (*vx++) + (*qx++);
          }
          /*
          Origin Q=Q'H:
            for (k = i; k < N; ++k) {
              tmp[y * N + x] += q[y * N + k] * ((k == x) - v[k] * v[x] * 2);
            }
          */
        }
      }
      //end_roi();
    }

    for (y = 0; y < N; ++y) for (x = i; x < N; ++x) {
      q[y * N + x] = tmp[y * N + x];
      tmp[y * N + x] = 0;
    }

    {
      //begin_roi();
      DTYPE *vy = v, *vk, *rk;
      for (y = i; y < N; ++y) {
        for (x = i; x < N; ++x) {
          if ((i + x * N) & 1) {
            tmp[y * N + x] =
              r[y + x * N] - 2 * (*vy) * (sb_dot(vv, r + i + x * N + 1, N - i - 1) + (*v) * r[i + x * N]);
          } else {
            tmp[y * N + x] = r[y + x * N] - 2 * (*vy) * sb_dot(v, r + i + x * N, N - i);
          }
          /*
          Origin R=HR':
            for (k = i; k < N; ++k)
              tmp[y * N + x] += h(y, k) * r[k * N + x];
          */
        }
        ++vy;
      }
      //end_roi();
    }

    for (y = i; y < N; ++y) for (x = i; x < N; ++x) {
      r[x * N + y] = tmp[y * N + x];
#ifdef DEBUG
      tmp[y * N + x] = 0;
#endif
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
    for (j = 0; j < N; ++j)
      r[i * N + j] = tmp[i * N + j];
  free(v);
  free(vv);
}
#undef h

