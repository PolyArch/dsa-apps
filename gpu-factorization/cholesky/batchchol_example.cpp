/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include batchchol_example.cpp 
 *   g++ -o a.out batchchol_example.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
 */
#include "../common/fileop.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "../../common/include/sim_timing.h"

#define _N_ 12
#define _batch_size_ 8

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

cuFloatComplex * cuComplexMatMul(int m, int n, const cuFloatComplex * A, const cuFloatComplex * B){
    cuFloatComplex * mul = (cuFloatComplex *) malloc(sizeof(cuFloatComplex) * m * n);
    for (int row = 0 ; row < m ; row ++){
    	for (int col = 0; col < n ; col ++){
	    cuFloatComplex temp = make_cuComplex(0.0, 0.0);

	    for (int col_A = 0; col_A < n ; col_A ++){
		//printf("First = %f + %f j, Next = %f + %f j\n",A[row + col_A * m].x, A[row + col_A * m].y, B[col_A + col * m].x, B[col_A + col * m].y);
		temp = cuCaddf(temp,cuCmulf(A[row + col_A * m],B[col_A + col * m]));
	    }
            //printf("Prod(%d,%d) = %f + %f j\n",row+1,col+1,temp.x,temp.y);
	    mul[row + col * m] = temp;
	}
    }
    //printMatrix(m,n,mul,m,"Product");
    return mul;
}

int main(int argc, char*argv[])
{

    FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
    if (!input_data || !ref_data) {
        puts("Data error!");
    return 1;
    }

    cuFloatComplex A0 [ _N_ * _N_ ];
    cuFloatComplex B0 [ _N_ ];

    read_n_float_complex(input_data, _N_ * _N_, A0);

    for (int row = 0; row < _N_ ; row++){
        B0[row] = make_cuFloatComplex((row+1)*1.0,(row+1)*1.0);
    }

    cusolverDnHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;

    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    const int batchSize = _batch_size_;
    const int nrhs = 1;
    const int m = _N_;
    const int lda = m;
    const int ldb = m;

    // Host Variables
    int infoArray_c[batchSize];
    cuFloatComplex L0_c[lda*m];
    cuFloatComplex *Aarray_c[batchSize];
    cuFloatComplex *Barray_c[batchSize];
    // GPU Variables
    cuFloatComplex **d_Aarray_c = NULL;
    cuFloatComplex **d_Barray_c = NULL;
    int *d_infoArray_c = NULL;

/* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&handle);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(handle, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: copy A to device */

    // Arrange Value Array Space in GPU
    for(int j = 0 ; j < batchSize ; j++){
        cudaStat1 = cudaMalloc ((void**)&Aarray_c[j], sizeof(cuFloatComplex) * lda * m);
        cudaStat2 = cudaMalloc ((void**)&Barray_c[j], sizeof(cuFloatComplex) * m * nrhs);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
    }
    // Arrange Info Array Space in GPU
    cudaStat1 = cudaMalloc ((void**)&d_infoArray_c, sizeof(int)*batchSize);
    assert(cudaSuccess == cudaStat1);

    // Arrange Batch Array Space in GPU
    cudaStat1 = cudaMalloc ((void**)&d_Aarray_c, sizeof(cuFloatComplex*) * batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_Barray_c, sizeof(cuFloatComplex*) * batchSize);
    assert(cudaSuccess == cudaStat1);

    // Copy Array A to each batch
    // Complex
    for (int idx_batch = 0 ;idx_batch < batchSize; idx_batch ++){
        cudaStat1 = cudaMemcpy(Aarray_c[idx_batch], A0, sizeof(cuFloatComplex) * lda * m, cudaMemcpyHostToDevice);
        cudaStat2 = cudaMemcpy(Barray_c[idx_batch], B0, sizeof(cuFloatComplex) * m * nrhs, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
    }

    // Copy Array A to batch array
    // Complex 
    cudaStat1 = cudaMemcpy(d_Aarray_c, Aarray_c, sizeof(cuFloatComplex*)*batchSize, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_Barray_c, Barray_c, sizeof(cuFloatComplex*)*batchSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    // Sync
    cudaDeviceSynchronize();

/* step 3: Cholesky factorization */
    // Solve
    // Complex
    printf("Cholesky Time = \n");
    begin_roi(); // Start Tick
    status = cusolverDnCpotrfBatched(
        handle,
        uplo,
        m,
        d_Aarray_c,
        lda,
        d_infoArray_c,
        batchSize);
    cudaStat1 = cudaDeviceSynchronize();
    end_roi(); // End Tick
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    // Copy Info Array Back
    // Complex
    cudaStat1 = cudaMemcpy(infoArray_c, d_infoArray_c, sizeof(int)*batchSize, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(L0_c, Aarray_c[_batch_size_ / 2], sizeof(cuFloatComplex) * lda * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    // Print Info Array
    // Complex
    for(int j = 0 ; j < batchSize ; j++){
        if(infoArray_c[j] != 0){
	    printf("Batch[%d] is not Positive Definite", j);
	}
    }

    // Clear for  Upper Triangular
    for (int row = 0; row < m; row++){
        for (int col = row + 1; col < m; col++){
            L0_c[row + col * m] = make_cuComplex(0.0,0.0);
        }
    }
    // reverse the L0
    for (int row = 0; row < m; row++){
        for (int col = row + 1; col < m; col++){
            cuFloatComplex temp = L0_c[row + col * m];
	    L0_c[row + col * m] = L0_c[col + row * m];
	    L0_c[col + row * m] = temp;
        }
    }

    // Solve Ax = b
    printf("Solve Ax=b Time = \n");
    begin_roi();
    status = cusolverDnCpotrsBatched(
        handle,
        uplo,
        m,
        nrhs, /* only support rhs = 1*/
        d_Aarray_c,
        lda,
        d_Barray_c,
        ldb,
        d_infoArray_c,
        batchSize);
    cudaStat1 = cudaDeviceSynchronize();
    end_roi();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(B0, Barray_c[_batch_size_ / 2], sizeof(cuFloatComplex) * m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(infoArray_c, d_infoArray_c, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);

    for (int i = 0;i < batchSize ; i++){
        if(infoArray_c[i] !=0){
            printf("batch %d is not working\n", i);
        }
    }

    // Print Result
    // Complex
    /*
    printMatrix(m,m,L0_c,lda,"L0");
    printf("=====\n");
    printMatrix(m, 1, B0, lda, "X");
    printf("=====\n");
   */ 

    // Test for Ref Data
    //assert(compare_n_float_complex(ref_data, _N_ * _N_, L0_c));


/* free resources */
    // Complex
    if (d_Aarray_c    ) cudaFree(d_Aarray_c);
    if (d_infoArray_c ) cudaFree(d_infoArray_c);
    if (d_Barray_c    ) cudaFree(d_Barray_c);

    if (handle      ) cusolverDnDestroy(handle);
    if (stream      ) cudaStreamDestroy(stream);

    cudaDeviceReset();

    return 0;
}
