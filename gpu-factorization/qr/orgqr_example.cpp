/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include orgqr_example.cpp 
 *   g++ -fopenmp -o a.out orgqr_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "../common/fileop.h"
#include "../../common/include/sim_timing.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#define _N_ 12

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
    const int m = _N_ ;
    const int n = _N_ ;
    const int lda = m;
/*       | 1 2  |
 *   A = | 4 5  |
 *       | 2 1  |
 */

    FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
    if (!input_data || !ref_data) {
        puts("Data error!");
    return 1;
    }

    cuFloatComplex A [ _N_ * _N_ ];

    read_n_float_complex(input_data, _N_ * _N_, A);

    cuFloatComplex Q[lda*n]; // orthonormal columns
    cuFloatComplex R[n*n]; // R = I - Q**T*Q 

    cuFloatComplex *d_A = NULL;
    cuFloatComplex *d_tau = NULL;
    int *devInfo = NULL;
    cuFloatComplex *d_work = NULL;

    cuFloatComplex *d_R = NULL;

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    int lwork = 0;

    int info_gpu = 0;

    const double h_one = 1;
    const cuFloatComplex h_minus_one = make_cuComplex(-1.0,-1.0);

    printf("A = (matlab base-1)\n");
    printMatrix(m, n, A, lda, "A");
    printf("=====\n");

// step 1: create cusolverDn/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(cuFloatComplex)*lda*n);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(cuFloatComplex)*n);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaStat4 = cudaMalloc ((void**)&d_R  , sizeof(cuFloatComplex)*n*n);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuFloatComplex)*lda*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
 
// step 3: query working space of geqrf and orgqr
    cusolver_status = cusolverDnCgeqrf_bufferSize(
        cusolverH,
        m,
        n,
        d_A,
        lda,
        &lwork_geqrf);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cusolver_status = cusolverDnCungqr_bufferSize(
        cusolverH,
        m,
        n,
        n,
        d_A,
        lda,
	d_tau,
        &lwork_orgqr);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
// lwork = max(lwork_geqrf, lwork_orgqr)
    lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
    cusolver_status = cusolverDnCgeqrf(
        cusolverH,
        m,
        n,
        d_A,
        lda,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is successful or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after geqrf: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

// step 5: compute Q
    cusolver_status= cusolverDnCungqr(
        cusolverH,
        m,
        n,
        n,
        d_A,
        lda,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
 
    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after orgqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    cudaStat1 = cudaMemcpy(Q, d_A, sizeof(cuFloatComplex)*lda*n, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("Q = (matlab base-1)\n");
    printMatrix(m, n, Q, lda, "Q");

// step 6: measure R = I - Q**T*Q
    memset(R, 0, sizeof(cuFloatComplex)*n*n);
    for(int j = 0 ; j < n ; j++){
        R[j + n*j] = make_cuComplex(1.0,1.0); // R(j,j)=1
    }

    cudaStat1 = cudaMemcpy(d_R, R, sizeof(cuFloatComplex)*n*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    // R = -Q**T*Q + I
    cublas_status = cublasCgemm_v2(
        cublasH,
        CUBLAS_OP_T, // Q**T
        CUBLAS_OP_N, // Q
        n, // number of rows of R
        n, // number of columns of R
        m, // number of columns of Q**T 
        &h_minus_one, /* host pointer */
        d_A, // Q**T
        lda,
        d_A, // Q
        lda,
        &h_one, /* hostpointer */
        d_R,
        n);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    double dR_nrm2 = 0.0;
    cublas_status = cublasCnrm2_v2(
        cublasH, n*n, d_R, 1, &dR_nrm2);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    printf("|I - Q**T*Q| = %E\n", dR_nrm2);

// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (d_R    ) cudaFree(d_R);

    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();

    return 0;
}
