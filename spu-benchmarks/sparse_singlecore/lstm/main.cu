#include <cstdio>
#include <stdint.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

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


__global__
void c_next(float *a, float *b, float *c, float *d, float *e) {
  int i = threadIdx.x;
  a[i] = 1.0 / (1.0 + exp(a[i]));
  c[i] = 1.0 / (1.0 + exp(c[i]));
  d[i] = 1.0 / (1.0 + exp(d[i]));
  e[i] = a[i] * b[i] + c[i] * d[i];
}

__global__
void h_next(float *a, float *b, float *c) {
  int i = blockIdx.x;
  c[i] = a[i] * tanh(b[i]);
}

__global__
void exp(float *a, float *sum) {
  int i = blockIdx.x;
  a[i] = exp(a[i]);
  sum[0] += a[i];
}

__global__
void div(float *a, float *sum) {
  int i = threadIdx.x;
  a[i] /= sum[0];
}

int main(int argc, char **argv) {
  int N;
  if (argc != 2) {
    N = 1024;
  } else {
    N = atoi(argv[1]);
  }

  cublasHandle_t cu_handle;
  cublasCreate(&cu_handle);

  float *x;
  cudaMalloc(&x, N * sizeof(float));

  float *w_f, *b_f, *_f;
  cudaMalloc(&w_f, N * N / 2 * sizeof(float));
  cudaMalloc(&b_f, N / 2 * sizeof(float));
  cudaMalloc(&_f, N / 2 * sizeof(float) * batch);

  float *w_i, *b_i, *_i;
  cudaMalloc(&w_i, N * N / 2 * sizeof(float));
  cudaMalloc(&b_i, N / 2 * sizeof(float));
  cudaMalloc(&_i, N / 2 * sizeof(float) * batch);

  float *w_c, *b_c, *_c;
  cudaMalloc(&w_c, N * N / 2 * sizeof(float));
  cudaMalloc(&b_c, N / 2 * sizeof(float));
  cudaMalloc(&_c, N / 2 * sizeof(float) * batch);

  float *w_o, *b_o, *_o;
  cudaMalloc(&w_o, N * N / 2 * sizeof(float));
  cudaMalloc(&b_o, N / 2 * sizeof(float));
  cudaMalloc(&_o, N / 2 * sizeof(float) * batch);

  float *w_y, *b_y, *_y;
  cudaMalloc(&w_y, N * N / 2 * sizeof(float));
  cudaMalloc(&b_y, N / 2 * sizeof(float));
  cudaMalloc(&_y, N / 2 * sizeof(float) * batch);

  float *c_nxt;
  cudaMalloc(&c_nxt, N / 2 * sizeof(float) * batch);

  float *h_nxt;
  cudaMalloc(&h_nxt, N / 2 * sizeof(float) * batch);

  float *sum;
  cudaMalloc(&sum, sizeof(float));

  float alpha = 1.0;

  cublasGemmEx(
    cu_handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    N / 2, N, 1,
    &alpha, w_f, CUDA_R_32F, N,
    x, CUDA_R_32F, 1,
    &alpha, b_f, CUDA_R_32F, N,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT
  );

  cudaMemcpy(_f, b_f, sizeof(float) * N / 2, cudaMemcpyDefault);

  c_next<<<1, N / 2>>>(_f, x, _i, _c, c_nxt);
  exp<<<1, N / 2>>>(_y, sum);
  div<<<1, N / 2>>>(_y, sum);

  begin_roi();

  for (int i = 0; i < batch; ++i) {
    cudaMemcpy(_f + i * N / 2, b_f, sizeof(float) * N / 2, cudaMemcpyDefault);
    cudaMemcpy(_i + i * N / 2, b_i, sizeof(float) * N / 2, cudaMemcpyDefault);
    cudaMemcpy(_c + i * N / 2, b_c, sizeof(float) * N / 2, cudaMemcpyDefault);
    cudaMemcpy(_o + i * N / 2, b_o, sizeof(float) * N / 2, cudaMemcpyDefault);
    cudaMemcpy(_y + i * N / 2, b_y, sizeof(float) * N / 2, cudaMemcpyDefault);
  }

#define cublas_mv(m, a, b) \
  cublasGemmEx( \
    cu_handle, \
    CUBLAS_OP_N, \
    CUBLAS_OP_N, \
    N / 2, N, batch, \
    &alpha, m, CUDA_R_32F, N, \
    a, CUDA_R_32F, batch, \
    &alpha, b, CUDA_R_32F, N, \
    CUDA_R_32F, \
    CUBLAS_GEMM_DEFAULT \
  )

  cublas_mv(w_f, x, _f);
  cublas_mv(w_i, x, _i);
  cublas_mv(w_c, x, _c);
  cublas_mv(w_o, x, _o);

  c_next<<<N / 2 * batch, 1>>>(_f, x, _i, _c, c_nxt);
  h_next<<<N / 2 * batch, 1>>>(_o, c_nxt, h_nxt);

  cublasGemmEx(
    cu_handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    N / 2, N / 2, batch,
    &alpha, w_y, CUDA_R_32F, N / 2,
    h_nxt, CUDA_R_32F, batch,
    &alpha, _y, CUDA_R_32F, N / 2,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT
  );

  exp<<<N / 2 * batch, 1>>>(_y, sum);
  div<<<N / 2 * batch, 1>>>(_y, sum);

  end_roi();

  cublasDestroy(cu_handle);

  return 0;
}
