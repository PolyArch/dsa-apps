#include <cstdio>
#include <stdint.h>
#include "mkl.h"
#include <cstring>

#include <math.h>

#define batch 1

#include <sys/time.h>

static __inline__ uint64_t rdtsc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static uint64_t ticks;

static void begin_roi() {

  ticks=rdtsc();

}


static void end_roi()   {

  ticks=(rdtsc()-ticks);
  printf("ticks: %lu\n", ticks);

}


void c_next(int N, float *a, float *b, float *c, float *d, float *e) {
  for (int i = 0; i < N / 2; ++i) {
    a[i] = 1.0 / (1.0 + exp(a[i]));
    c[i] = 1.0 / (1.0 + exp(c[i]));
    d[i] = 1.0 / (1.0 + exp(d[i]));
    e[i] = a[i] * b[i] + c[i] * d[i];
  }
}

void h_next(int N, float *a, float *b, float *c) {
  for (int i = 0; i < N / 2; ++i) {
    c[i] = a[i] * tanh(b[i]);
  }
}

void exp(int N, float *a, float *sum) {
  for (int i = 0; i < N / 2; ++i) {
    a[i] = exp(a[i]);
    sum[0] += a[i];
  }
}

void div(int N, float *a, float *sum) {
  for (int i = 0; i < N / 2; ++i) {
    a[i] /= sum[0];
  }
}

int main(int argc, char **argv) {
  //mkl_set_num_threads_local(8);
  int N;
  if (argc != 2) {
    N = 1024;
  } else {
    N = atoi(argv[1]);
  }

  float *x;
  x = (float*) malloc(N * sizeof(float));

  float *w_f, *b_f, *_f;
  w_f = (float*)malloc(N * N / 2 * sizeof(float));
  b_f = (float*)malloc(N / 2 * sizeof(float));
  _f  = (float*)malloc(N / 2 * sizeof(float) * batch);

  float *w_i, *b_i, *_i;
  w_i = (float*)malloc(N * N / 2 * sizeof(float));
  b_i = (float*)malloc(N / 2 * sizeof(float));
  _i  = (float*)malloc(N / 2 * sizeof(float) * batch);

  float *w_c, *b_c, *_c;
  w_c = (float*)malloc(N * N / 2 * sizeof(float));
  b_c = (float*)malloc(N / 2 * sizeof(float));
  _c  = (float*)malloc(N / 2 * sizeof(float) * batch);

  float *w_o, *b_o, *_o;
  w_o = (float*)malloc(N * N / 2 * sizeof(float));
  b_o = (float*)malloc(N / 2 * sizeof(float));
  _o  = (float*)malloc(N / 2 * sizeof(float) * batch);

  float *w_y, *b_y, *_y;
  w_y = (float*)malloc(N * N / 2 * sizeof(float));
  b_y = (float*)malloc(N / 2 * sizeof(float));
  _y  = (float*)malloc(N / 2 * sizeof(float) * batch);

  float *c_nxt;
  c_nxt = (float*)malloc(N / 2 * sizeof(float) * batch);

  float *h_nxt;
  h_nxt = (float*)malloc(N / 2 * sizeof(float) * batch);

  float *sum;
  sum = (float*)malloc(sizeof(float));

  float alpha = 1.0;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    N / 2, 1, N,
    alpha, w_f, N,
    x, 1,
    alpha, b_f, 1
  );

  c_next(N, _f, x, _i, _c, c_nxt);
  h_next(N, _o, c_nxt, h_nxt);
  exp(N, _y, sum);
  div(N, _y, sum);

  begin_roi();

  memcpy(_f, b_f, N / 2 * sizeof(float));
  memcpy(_i, b_i, N / 2 * sizeof(float));
  memcpy(_o, b_o, N / 2 * sizeof(float));
  memcpy(_c, b_c, N / 2 * sizeof(float));
  memcpy(_y, b_y, N / 2 * sizeof(float));

  #define cublas_mv(m, a, b) \
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
    N / 2, 1, N, \
    alpha, m, N, \
    a, 1, \
    alpha, b, 1 \
  )

  cublas_mv(w_f, x, _f);
  cublas_mv(w_i, x, _i);
  cublas_mv(w_c, x, _c);
  cublas_mv(w_o, x, _o);

  c_next(N, _f, x, _i, _c, c_nxt);
  h_next(N, _o, c_nxt, h_nxt);

  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    N / 2, 1, N / 2,
    alpha, w_y, N / 2,
    h_nxt, 1,
    alpha, _y, 1
  );

  exp(N, _y, sum);
  div(N, _y, sum);

  end_roi();

  return 0;
}
