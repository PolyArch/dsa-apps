#include "qr.h"
#include "norm.h"
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
    //SB_CONFIG(norm_config, norm_size);
    SB_CONST(P_norm_carry, ri.b, 11);
    SB_DMA_READ(a, 16, 16, n / 4, P_norm_A);
    SB_DMA_READ(b, 16, 16, n / 4, P_norm_B);
    SB_RECURRENCE(P_norm_R, P_norm_carry, n / 4 - 11);
    SB_DMA_WRITE(P_norm_R, 8, 8, 11, sum);
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
    //SB_CONFIG(norm_config, norm_size);
    SB_CONST(P_norm_carry, ri.b, 1);
    SB_DMA_READ(a, 16, 16, n / 4, P_norm_A);
    SB_DMA_READ(b, 16, 16, n / 4, P_norm_B);
    SB_RECURRENCE(P_norm_R, P_norm_carry, n / 4 - 1);
    SB_DMA_WRITE(P_norm_R, 8, 8, 1, sum);
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
  SB_CONFIG(norm_config, norm_size);
  int i, j, k, x, y; DTYPE *tmp = (DTYPE *) malloc(N * N * sizeof(DTYPE));
  DTYPE *v = (DTYPE *) malloc(N * sizeof(DTYPE)),
        *vv= (DTYPE *) malloc(N * sizeof(DTYPE)),
        *rr= (DTYPE *) malloc(N * sizeof(DTYPE));
  for (i = 0; i < N; ++i) {
    q[i * (N + 1)] = 1;
  }
  for (i = 0; i < N * N; ++i) {
    r[i] = a[i];
  }
  for (i = 0; i < N; ++i) {
    float dot = 0.;

    {
      DTYPE *rp = r + i * (N + 1), *vp = v;
      for (j = i; j < N; ++j) {
        *vp = *rp;
        dot += *vp * *vp;
        rp += N;
        ++vp;
      }
    }

    *v += (r[i * (N + 1)] < -eps ? -1 : 1) * sqrt(dot);

    /*
    dot = 0.;
    for (j = 0; j < N - i; ++j)
      dot += v[j] * v[j];
    */
    dot = sb_dot(v, v, N - i);

    dot = sqrt(dot);
    for (j = 0; j < N - i; ++j) {
      v[j] /= dot;
      if (j) {
        vv[j - 1] = v[j];
      }
    }

    for (y = 0; y < N; ++y) for (x = i; x < N; ++x) tmp[y * N + x] = 0;

    {
      DTYPE *vk, *vx, *qk, *qx;
      for (y = 0; y < N; ++y) {
        for (x = i, vx = v, qx = q + y * N + i; x < N; ++x) {
          //Sequential pattern:
          if (((y * N + i) & 1) || N - i < 4) {
            if (N - i >= 5) {
              tmp[y * N + x] =
                (q[y * N + i] * (*v) + sb_dot(q + y * N + i + 1, vv, N - i - 1)) * -2 * (*vx++) + (*qx++);
            } else {
              for (k = i, vk = v, qk = q + y * N + i; k < N; ++k) {
                tmp[y * N + x] += (*qk++) * (*vk++);
              }
              tmp[y * N + x] *= -2 * (*vx++);
              tmp[y * N + x] += *qx++;
            }
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
    }

    for (y = 0; y < N; ++y) for (x = i; x < N; ++x) {
      q[y * N + x] = tmp[y * N + x];
      tmp[y * N + x] = 0;
    }

    {
      DTYPE *vy = v, *vk, *rk, *head_rr;
      for (y = i; y < N; ++y) {
        for (x = i; x < N; ++x) {
          for (k = i, vk = v, rk = r + i * N + x; k < N; ++k) {
            tmp[y * N + x] += (*vk++) * (*rk);
            rk += N;
          }
          tmp[y * N + x] = r[y * N + x] - 2 * (*vy) * tmp[y * N + x];
          /*
          Origin R=HR':
            for (k = i; k < N; ++k)
              tmp[y * N + x] += h(y, k) * r[k * N + x];
          */
        }
        ++vy;
      }
    }

    for (y = i; y < N; ++y) for (x = i; x < N; ++x) {
      r[y * N + x] = tmp[y * N + x];
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
  free(v);
  free(vv);
  free(rr);
}
#undef h

