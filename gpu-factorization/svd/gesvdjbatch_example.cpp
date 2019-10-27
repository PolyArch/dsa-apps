/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include gesvdjbatch_example.cpp 
 *   g++ -o gesvdjbatch_example gesvdjbatch_example.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "../common/fileop.h"
#include "../../common/include/sim_timing.h"

#ifndef _N_
#define _N_ 32
#endif

#ifndef _batch_size_

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif

#define _batch_size_ BATCH_SIZE
#endif

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %20.16E\n", name, row+1, col+1, Areg);
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


int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    const int m = _N_; /* 1 <= m <= 32 */
    const int n = _N_; /* 1 <= n <= 32 */
    const int lda = m; /* lda >= m */
    const int ldu = m; /* ldu >= m */
    const int ldv = n; /* ldv >= n */
    const int batchSize = _batch_size_;
    const int minmn = (m < n)? m : n; /* min(m,n) */
/*  
 *        |  1  -1  |
 *   A0 = | -1   2  |
 *        |  0   0  |
 *
 *   A0 = U0 * S0 * V0**T
 *   S0 = diag(2.6180, 0.382) 
 *
 *        |  3   4  |
 *   A1 = |  4   7  |
 *        |  0   0  |
 *
 *   A1 = U1 * S1 * V1**T
 *   S1 = diag(9.4721, 0.5279) 
 */
 
    cuFloatComplex A[lda*n*batchSize]; /* A = [A0 ; A1] */
    cuFloatComplex U[ldu*m*batchSize]; /* U = [U0 ; U1] */
    cuFloatComplex V[ldv*n*batchSize]; /* V = [V0 ; V1] */
    float S[minmn*batchSize]; /* S = [S0 ; S1] */
    int info[batchSize];       /* info = [info0 ; info1] */

    cuFloatComplex *d_A  = NULL; /* lda-by-n-by-batchSize */
    cuFloatComplex *d_U  = NULL; /* ldu-by-m-by-batchSize */
    cuFloatComplex *d_V  = NULL; /* ldv-by-n-by-batchSize */
    float *d_S  = NULL; /* minmn-by-batchSizee */
    int* d_info  = NULL; /* batchSize */
    int lwork = 0;       /* size of workspace */
    cuFloatComplex *d_work = NULL; /* device workspace for gesvdjBatched */

    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const int sort_svd  = 0;   /* don't sort singular values */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

/* residual and executed_sweeps are not supported on gesvdjBatched */
    double residual = 0;
    int executed_sweeps = 0;

// Load Input Data
    FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
    if (!input_data || !ref_data) {
        puts("Data error!");
    return 1;
    }

    cuFloatComplex A_temp [ _N_ * _N_ ];

    read_n_float_complex(input_data, _N_ * _N_, A_temp);

    for(int idx_batch = 0; idx_batch < batchSize; idx_batch ++){
	for (int row = 0; row < m; row ++){
	    for (int col = 0; col < n; col ++){
	        A[row + col*lda + idx_batch * m * n] = A_temp[row + col*lda];
	    }
	}
    }
/*
    printf("example of gesvdjBatched \n");
    printf("m = %d, n = %d \n", m, n);
    printf("tol = %E, default value is machine zero \n", tol);
    printf("max. sweeps = %d, default value is 100\n", max_sweeps);

    printf("A_temp = (matlab base-1)\n");
    printMatrix(m, n, A_temp, lda, "A_temp");
    //printf("A[:,:,5] = (matlab base-1)\n");
    //printMatrix(m, n, A[4 * m * n], lda, "A[5]");
    printf("=====\n");
*/

/* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* disable sorting */
    status = cusolverDnXgesvdjSetSortEig(
        gesvdj_params,
        sort_svd);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 3: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(cuFloatComplex)*lda*n*batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_U   , sizeof(cuFloatComplex)*ldu*m*batchSize);
    cudaStat3 = cudaMalloc ((void**)&d_V   , sizeof(cuFloatComplex)*ldv*n*batchSize);
    cudaStat4 = cudaMalloc ((void**)&d_S   , sizeof(float)*minmn*batchSize);
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int   )*batchSize);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuFloatComplex)*lda*n*batchSize, cudaMemcpyHostToDevice);
    cudaStat2 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

/* step 4: query working space of gesvdjBatched */
    status = cusolverDnCgesvdjBatched_bufferSize(
        cusolverH,
        jobz,
        m,
        n,
        d_A,
        lda,
        d_S,
        d_U,
        ldu,
        d_V,
        ldv,
        &lwork,
        gesvdj_params,
        batchSize
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuFloatComplex)*lwork);
    assert(cudaSuccess == cudaStat1);

    printf("%d %d\n", _N_, _batch_size_);
    status = cusolverDnCgesvdjBatched(
        cusolverH,
        jobz,
        m,
        n,
        d_A,
        lda,
        d_S,
        d_U,
        ldu,
        d_V,
        ldv,
        d_work,
        lwork,
        d_info,
        gesvdj_params,
        batchSize
    );
/* step 5: compute singular values of A0 and A1 */
    begin_roi();
    for (int i = 0; i < 100; ++i) {
      status = cusolverDnCgesvdjBatched(
          cusolverH,
          jobz,
          m,
          n,
          d_A,
          lda,
          d_S,
          d_U,
          ldu,
          d_V,
          ldv,
          d_work,
          lwork,
          d_info,
          gesvdj_params,
          batchSize
      );
      cudaStat1 = cudaDeviceSynchronize();
    }
    assert(CUSOLVER_STATUS_SUCCESS == status);
    end_roi();
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(U    , d_U   , sizeof(cuFloatComplex)*ldu*m*batchSize, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V    , d_V   , sizeof(cuFloatComplex)*ldv*n*batchSize, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(S    , d_S   , sizeof(float)*minmn*batchSize, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(&info, d_info, sizeof(int) * batchSize       , cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
/*
    for(int i = 0 ; i < batchSize ; i++){
        if ( 0 == info[i] ){
            printf("matrix %d: gesvdj converges \n", i);
        }else if ( 0 > info[i] ){
// only info[0] shows if some input parameter is wrong.
// If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
 
            printf("Error: %d-th parameter is wrong \n", -info[i] );
            exit(1);
        }else { // info = m+1 
// if info[i] is not zero, Jacobi method does not converge at i-th matrix.
            printf("WARNING: matrix %d, info = %d : gesvdj does not converge \n", i, info[i] );
        }
    }
*/
/* Step 6: show singular values and singular vectors */
/*
    float *S0 = S;
    float *S1 = S + minmn;
    printf("==== \n");
    for(int i = 0 ; i < minmn ; i++){
        printf("S0(%d) = %f\n", i+1, S0[i]);
    }
    printf("==== \n");
    for(int i = 0 ; i < minmn ; i++){
        printf("S1(%d) = %f\n", i+1, S1[i]);
    }
    printf("==== \n");

    cuFloatComplex *U0 = U;
    cuFloatComplex *U1 = U + ldu*m; // Uj is m-by-m
    printf("U0 = (matlab base-1)\n");
    printMatrix(m, m, U0, ldu, "U0");
    printf("U1 = (matlab base-1)\n");
    printMatrix(m, m, U1, ldu, "U1");

    cuFloatComplex *V0 = V;
    cuFloatComplex *V1 = V + ldv*n; // Vj is n-by-n
    printf("V0 = (matlab base-1)\n");
    printMatrix(n, n, V0, ldv, "V0");
    printf("V1 = (matlab base-1)\n");
    printMatrix(n, n, V1, ldv, "V1");
*/
/*
 * The folowing two functions do not support batched version.
 * The error CUSOLVER_STATUS_NOT_SUPPORTED is returned. 
 */
    status = cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdj_params,
        &executed_sweeps);
    assert(CUSOLVER_STATUS_NOT_SUPPORTED == status);

    status = cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdj_params,
        &residual);
    assert(CUSOLVER_STATUS_NOT_SUPPORTED == status);

/* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_U    ) cudaFree(d_U);
    if (d_V    ) cudaFree(d_V);
    if (d_S    ) cudaFree(d_S);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

    cudaDeviceReset();

    return 0;
}
