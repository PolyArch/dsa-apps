/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This example demonstrates how to get better performance by
 * batching CUBLAS calls with the use of using streams
 */

#include "../../common/include/sim_timing.h"
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <float.h>
#endif

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

// Utilities and system includes
#include <helper_cuda.h>
#include "batchCUBLAS.h"

const char *sSDKname = "batchCUBLAS";

//============================================================================================
// Device information utilities
//============================================================================================


    //return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    //return cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);

#ifndef M
#define M 12
#endif

#ifndef N
#define N 16
#endif

#ifndef K
#define K 64
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif

int main(int argc, char *argv[])
{
  cuFloatComplex *a[BATCH_SIZE], *bufferA;
  cuFloatComplex *b[BATCH_SIZE], *bufferB;
  cuFloatComplex *c[BATCH_SIZE], *bufferC;

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaStream_t streams[BATCH_SIZE];

  assert(cudaSuccess == cudaMalloc(&bufferA, (BATCH_SIZE + 1) * M * K * sizeof(cuFloatComplex)));
  assert(cudaSuccess == cudaMalloc(&bufferB, (BATCH_SIZE + 1) * N * K * sizeof(cuFloatComplex)));
  assert(cudaSuccess == cudaMalloc(&bufferC, (BATCH_SIZE + 1) * M * N * sizeof(cuFloatComplex)));

  for (int i = 0; i < BATCH_SIZE; ++i) {
    a[i] = bufferA + i * M * K;
    b[i] = bufferB + i * N * K;
    c[i] = bufferC + i * N * M;
    cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking);
  }

  cuFloatComplex alpha[BATCH_SIZE];
  cuFloatComplex beta[BATCH_SIZE];

  std::cout << M << ", " << N << ", " << K << ", " << BATCH_SIZE << std::endl;
  begin_roi();
  for (int j = 0; j < 100; ++j) {
    for (int i = 0; i < BATCH_SIZE; ++i) {
      cublasSetStream(handle, streams[i]);
      assert(CUBLAS_STATUS_SUCCESS == cublasCgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha, a[0], M, b[0], K,
            beta, c[0], M));
    }
    assert(cudaSuccess == cudaDeviceSynchronize());
  }
  end_roi();


  assert(cudaSuccess == cudaFree(bufferA));
  assert(cudaSuccess == cudaFree(bufferB));
  assert(cudaSuccess == cudaFree(bufferC));
  cublasDestroy(handle);

  return 0;
}
