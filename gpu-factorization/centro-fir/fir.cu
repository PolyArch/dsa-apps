#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../../common/include/sim_timing.h"
#include <cuComplex.h>
#define THREAD_NUM 512

void init_arr(cuFloatComplex *arr, int len)
{
   int i = 0;;  
   for (i = 0; i < len; i++) {
   arr[i] = make_cuFloatComplex(0.0f,0.0f);
   }
}

void rand_arr(cuFloatComplex *arr, int len)
{
   int i = 0;
   for (i = 0; i < len; i++) {
   arr[i] = make_cuFloatComplex((rand()%1000) * 0.01,(rand()%1000)*0.01);
   }
}

double convolve(const cuFloatComplex *a, const cuFloatComplex *b, cuFloatComplex *d, int n)
{
   clock_t startCPU, endCPU;
   int i, j = 0;
   startCPU = clock();
   printf("CPU time = ");
   begin_roi();
   for (i = 0; i < n; i++)
       for (j = 0; j < n; j++) {
           d[i+j] = cuCaddf(d[i+j], cuCmulf(a[i], b[j]));
       }
       
   end_roi();
   endCPU = clock();
   return (double)(endCPU - startCPU);
}

__global__ static void ConvolveCUDA(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* c, int n)
{
   int i = 0;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

//Method 3    （improved from Method 2）
   if (idx < n)
   {
        cuFloatComplex t1 = make_cuFloatComplex(0.0,0.0); 
        cuFloatComplex t2 = make_cuFloatComplex(0.0,0.0);
       for (i = 0; i <= idx; i++) {
           t1 = cuCaddf(t1, cuCmulf(a[i], b[idx - i]));
       }
       for (i = idx+1; i < n; i++) {
           t2 = cuCaddf(t2, cuCmulf(a[i], b[n+idx-i]));
       }
       c[idx] = t1;
       c[n+idx] = t2;
   }

/*//Method 2
         for (i = 0; i < n; i++) {
           if (idx >= i) {
               t1 += a[i] * b[idx - i];
           }
           else {
               t2 += a[i] * b[n + idx - i];
           }
              c[idx] = t1;
           c[n+idx] = t2;
       }
*/


/*//Method 1
   if(idx < (2*n-1))
   {
       if (idx <= (n-1)){
           float t = 0;
           for (i = 0; i <= idx; i++) {
               t += a[i] * b[idx - i];
           }
           c[idx] = t;
       }
       if (idx > (n-1)){
           idx = 2*n-1 - idx -1;
           float t = 0;
           for (i = 0; i <= idx; i++) {
               t += a[(n - 1) - (idx - i)] * b[(n -1) - i];
           }
           c[2*n-1-idx -1] = t;
       }
   }
*/
}

double convolveCUDA(const cuFloatComplex *a, const cuFloatComplex *b, cuFloatComplex *c, int n)
{
   cuFloatComplex *a_d, *b_d, *c_d;
   clock_t start, end;
   int BLOCK_NUM = n / THREAD_NUM + ((n % THREAD_NUM > 0)?1:0);
   //int BLOCK_NUM = (n + THREAD_NUM) / THREAD_NUM;
   
   cudaMalloc((void**) &a_d, sizeof(cuFloatComplex) * n);
   cudaMalloc((void**) &b_d, sizeof(cuFloatComplex) * n);
   cudaMalloc((void**) &c_d, sizeof(cuFloatComplex) * (2*n-1));
   //printf("GPU time = ");
   //begin_roi();
   cudaMemcpy(a_d, a, sizeof(cuFloatComplex) * n, cudaMemcpyHostToDevice);
   cudaMemcpy(b_d, b, sizeof(cuFloatComplex) * n, cudaMemcpyHostToDevice);
   cudaMemcpy(c_d, c, sizeof(cuFloatComplex) * (2*n-1), cudaMemcpyHostToDevice);
   
   int repeat_time = 1000;
   begin_roi();
   for (int i=0;i < repeat_time;i++){
      ConvolveCUDA<<<BLOCK_NUM, THREAD_NUM>>>(a_d, b_d, c_d, n);    
   }
   end_roi();
   cudaMemcpy(c, c_d, sizeof(cuFloatComplex) * (2*n-1), cudaMemcpyDeviceToHost);
   //end_roi();
   
   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);
   
   return 0.0;
}

void compare_arr(const cuFloatComplex* a, const cuFloatComplex* b, int len)
{
    float max_err = 0;
    float average_err = 0;
    int i = 0;

    for(i = 0; i < len; i++) {
       if(cuCabsf(b[i]) != 0) {
           float err = cuCabsf(cuCdivf(cuCsubf(a[i],  b[i]), b[i]));
           if(max_err < err) max_err = err;
           average_err += err;
       }
    }

    printf("Max error: %g\tAverage error: %g\n", max_err, average_err / (len * len));
}

int main()
{
   cuFloatComplex *a, *b, *c, *d;
   int m, n = 0;

    printf("\nPlease input length n:");
    scanf("%d", &n);
   
   m = 2 * n - 1;
   
    a = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * n);
    b = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * n);
    c = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * m);
    d = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * m);

    srand((unsigned int)time(NULL) + rand());

    rand_arr(a, n);
    rand_arr(b, n);
   init_arr(c, m);
   init_arr(d, m);
   
   //for (int i = 0; i < n; i++) {printf("a[%d] = %.2f\t\tb[%d] = %.2f\n", i, a[i], i, b[i]);}
    //printf("\n");

    clock_t timeGPU = convolveCUDA(a, b, c, n);

   //clock_t timeCPU = convolve(a, b, d, n);
   clock_t timeCPU = 1;
   
    compare_arr(c, d, m);

    double secGPU = (double) timeGPU / CLOCKS_PER_SEC;
   double secCPU = (double) timeCPU / CLOCKS_PER_SEC;
   float ratio = secCPU / secGPU;
    printf("CPU vs GPU Time used: %.2f  vs  %.2f\n", secCPU, secGPU);
   printf("CPU vs GPU ratio: %.2f\n\n", ratio);    
   
   //for (int j = 0; j < m; j++) {printf("c[%d] = %.2f\t\td[%d] = %.2f\n", j, c[j], j, d[j]);}
    //printf("\n");
   
   free(a);
   free(b);
   free(c);
   free(d);

}
