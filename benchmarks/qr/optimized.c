#include "qr.h"
#include "../../common/include/sim_timing.h"

#define h(x, y) (((x) >= i) && ((y) >= i) ? (((x) == (y)) - v[x] * v[y] * 2.) : (x) == (y))
void qr(float *a, float *q, float *r) {
  int i, j, k, x, y; float *tmp = (float *) malloc(N * N * sizeof(float));
  float *v = (float *) malloc(N * sizeof(float));
  for (i = 0; i < N; ++i) {
    q[i * (N + 1)] = 1;
  }
  for (i = 0; i < N * N; ++i) {
    r[i] = a[i];
  }
  for (i = 0; i < N; ++i) {
    float dot = 0.;

    {
      float *vp = v + i, *rp = r + (N + 1) * i;
      for (j = i; j < N; ++j) {
        *vp = *rp;
        dot += *vp * *vp;
        ++vp;
        rp += N;
      }
    }

    v[i] += (r[i * (N + 1)] < -eps ? -1 : 1) * sqrt(dot);

    dot = 0.;
    for (j = i; j < N; ++j)
      dot += v[j] * v[j];

    dot = sqrt(dot);
    for (j = i; j < N; ++j)
      v[j] /= dot;

    for (y = 0; y < N; ++y) for (x = i; x < N; ++x) tmp[y * N + x] = 0;

    {
      float *vk, *vx, *qk, *qx;
      for (y = 0; y < N; ++y) {
        for (x = i, vx = v + i, qx = q + y * N + i; x < N; ++x) {
          //Sequential pattern:
          for (k = i, vk = v + i, qk = q + y * N + i; k < N; ++k) {
            tmp[y * N + x] += (*qk++) * (*vk++);
          }
          tmp[y * N + x] = tmp[y * N + x] * -2 * (*vx++) + (*qx++);
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
      float *vy = v + i, *vk, *rk;
      for (y = i; y < N; ++y) {
        for (x = i; x < N; ++x) {
          rk = r + i * N + x;
          for (k = i, vk = v + i; k < N; ++k) {
            tmp[y * N + x] += (*vk++) * (*rk);
            rk += N;
          }
          tmp[y * N + x] *= -2. * (*vy);
          tmp[y * N + x] += r[y * N + x];
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
}
#undef h

