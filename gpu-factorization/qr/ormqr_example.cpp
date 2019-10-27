/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include ormqr_example.cpp 
 *   nvcc -o -fopenmp a.out ormqr_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../common/fileop.h"
#include "../../common/include/sim_timing.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#ifndef _N_
#define _N_          32
#endif

#ifndef _BATCH_SIZE_
#define _BATCH_SIZE_ 8
#endif

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void printMatrix(int m, int n, const cuFloatComplex * A, int lda, const char* name)
{
    for (int row = 0 ; row < m ; row ++){
        for (int col = 0 ; col < n ; col++){
            cuFloatComplex A_c = A[row + col * lda];
            printf("%s(%d,%d) = %f + %f j\n",name, row +1, col +1, A_c.x, A_c.y);
        }
    }
}

cuFloatComplex A [ _N_ * _N_ ];

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    cudaStream_t streams[_BATCH_SIZE_];

    for (int i = 0; i < _BATCH_SIZE_; ++i) {
      cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking);
    }

    const int m = _N_ ;
    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors

    FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
    if (!input_data || !ref_data) {
        puts("Data error!");
    return 1;
    }

    read_n_float_complex(input_data, _N_ * _N_, A);

    cuFloatComplex *d_A[_BATCH_SIZE_]; // linear memory of GPU  
    cuFloatComplex *d_tau[_BATCH_SIZE_]; // linear memory of GPU 
    int *devInfo[_BATCH_SIZE_]; // info in gpu (device copy)
    cuFloatComplex *d_work[_BATCH_SIZE_];
    int  lwork = 0; 

    int info_gpu = 0;

    //printf("A = (matlab base-1)\n");
    //printMatrix(m, m, A, lda, "A");
    //printf("=====\n");

// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    for (int i = 0; i < _BATCH_SIZE_; ++i) {
    
// step 2: copy A and B to device
      cudaStat1 = cudaMalloc((void**)&d_A[i]  , sizeof(cuFloatComplex) * lda * m);
      cudaStat2 = cudaMalloc((void**)&d_tau[i], sizeof(cuFloatComplex) * m);
      cudaStat3 = cudaMalloc((void**)&devInfo[i], sizeof(int));
      assert(cudaSuccess == cudaStat1);
      assert(cudaSuccess == cudaStat2);
      assert(cudaSuccess == cudaStat3);

      cudaStat1 = cudaMemcpy(d_A[i], A, sizeof(cuFloatComplex) * lda * m   , cudaMemcpyHostToDevice);
      assert(cudaSuccess == cudaStat1);
    }
// step 3: query working space of geqrf and ormqr
    //begin_roi();
    cusolver_status = cusolverDnCgeqrf_bufferSize(
        cusolverH, 
        m, 
        m, 
        d_A[0], 
        lda,
        &lwork);
    //assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
 
    for (int i = 0; i < _BATCH_SIZE_; ++i)
      cudaStat1 = cudaMalloc(d_work + i, sizeof(cuFloatComplex)*lwork);
    //assert(cudaSuccess == cudaStat1);

    cusolver_status = cusolverDnCgeqrf(
        cusolverH, 
        m, 
        m, 
        d_A[0], 
        lda, 
        d_tau[0], 
        d_work[0], 
        lwork,
        devInfo[0]);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    // step 4: compute QR factorization
    begin_roi();
    for (int j = 0; j < 100; ++j) {
      for (int i = 0; i < _BATCH_SIZE_; ++i) {
        cusolverDnSetStream(cusolverH, streams[i]);
        cusolver_status = cusolverDnCgeqrf(
            cusolverH, 
            m, 
            m, 
            d_A[0], 
            lda, 
            d_tau[0], 
            d_work[0], 
            lwork, 
            devInfo[0]);
      }
      assert(cudaDeviceSynchronize() == cudaSuccess);
    }
    end_roi();

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    for (int i = 0; i < _BATCH_SIZE_; ++i) {
      // check if QR is good or not
      cudaStat1 = cudaMemcpy(&info_gpu, devInfo[i], sizeof(int), cudaMemcpyDeviceToHost);
      assert(cudaSuccess == cudaStat1);
      assert(0 == info_gpu);
    }

    //printf("after geqrf: info_gpu = %d\n", info_gpu);

// free resources
  for (int i = 0; i < _BATCH_SIZE_; ++i) {
    cudaFree(d_A[i]);
    cudaFree(d_tau[i]);
    cudaFree(d_work[i]);
    cudaFree(devInfo[i]);
  }


    if (cublasH ) cublasDestroy(cublasH);   
    if (cusolverH) cusolverDnDestroy(cusolverH);   

    cudaDeviceReset();

    return 0;
}

