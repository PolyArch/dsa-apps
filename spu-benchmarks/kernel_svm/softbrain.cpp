#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <inttypes.h>
#include "matrix_mul.dfg.h"
#include "svm_func.dfg.h"
#include "dot_prod.dfg.h"
#include "norm.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
using namespace std;

#define VTYPE double
#define C 0.1
#define tol 0.02
#define max_passes 100
#define M 52 // number of instances
#define N 52 // number of features

std::pair<VTYPE,VTYPE> *gram_mat_pair;
int *gram_ptr;

float min(float a, float b){
    return a<b?a:b;
}
float max(float a, float b){
    return a>b?a:b;
}

void mm(std::pair<VTYPE,VTYPE> *matrix1, int *row_ptr1, std::pair<VTYPE,VTYPE> *matrix2, int *col_ptr2, int nnz1, int nnz2){
  // redundant calculation of symmetric matrix for now
  // printf("gram matrix calculation starts\n");
  gram_mat_pair = (std::pair<VTYPE,VTYPE>*)malloc(nnz1*nnz2*sizeof(std::pair<VTYPE,VTYPE>));
  gram_ptr = (int*)malloc((M+1)*sizeof(int)); // equal to the number of rows

  int last = -1;
  int ptr1, ptr2, end1, end2;
  // int accum = 0;
  int nrows1 = M, ncols2 = M;


  SB_CONFIG(matrix_mul_config,matrix_mul_size);

  for (int i=0; i<nrows1; i++){
    gram_ptr[i] = -1; 

    // SB_2D_CONST(P_matrix_mul_done,2,ncols2-1,0,1,1);
    // i need to reset the accumulator
    SB_2D_CONST(P_matrix_mul_done,2,ncols2-1,1,1,1);
    ptr1 = row_ptr1[i];
    end1 = row_ptr1[i+1];

    SB_DMA_WRITE(P_matrix_mul_Index, 2*sizeof(VTYPE), sizeof(VTYPE), 10000000, &gram_mat_pair[last].first);
    SB_DMA_WRITE(P_matrix_mul_Val, 2*sizeof(VTYPE), sizeof(VTYPE), 10000000, &gram_mat_pair[last].second);

    for (int j=0; j<ncols2; j++){
      ptr2 = col_ptr2[j];
      end2 = col_ptr2[j+1];
      SB_DMA_READ(&(matrix1[ptr1].first), 2*sizeof(VTYPE), sizeof(VTYPE), end1-ptr1, P_matrix_mul_A);
      SB_DMA_READ(&(matrix2[ptr2].first), 2*sizeof(VTYPE), sizeof(VTYPE), end2-ptr2, P_matrix_mul_B);
      SB_DMA_READ(&(matrix1[ptr1].second), 2*sizeof(VTYPE), sizeof(VTYPE), end1-ptr1, P_matrix_mul_C);
      SB_DMA_READ(&(matrix2[ptr2].second), 2*sizeof(VTYPE), sizeof(VTYPE), end2-ptr2, P_matrix_mul_D);
      SB_CONST(P_matrix_mul_A, SENTINAL, 1);
      SB_CONST(P_matrix_mul_B, SENTINAL, 1);
      SB_CONST(P_matrix_mul_C, 0, 1);
      SB_CONST(P_matrix_mul_D, 0, 1);
      SB_CONST(P_matrix_mul_I, j, 1);
    }

    uint64_t nz_count;
    SB_RECV(P_matrix_mul_nz_count,nz_count);
    // wait on this output?: continue when 'only' this output is available
    SB_RESET();
    SB_WAIT_ALL(); 
    last+=nz_count;
    // printf("nz_count: %ld\n", nz_count);
    gram_ptr[i]=last;
  }
  gram_ptr[M] = last;
  /*
  for(int i=0; i<=M; ++i){
    printf("value of gram_ptr is: %d\n", gram_ptr[i]);
  }
  */
  // printf("gram matrix calculation done\n");
}

// this is like both are sparse- need different definition: I need an output
// even if 0
VTYPE dot_prod_sparse(std::pair<VTYPE,VTYPE> *data, int ptr1, int end1, int ptr2, int end2){
  VTYPE accum = 0;

  SB_CONFIG(dot_prod_config, dot_prod_size);
  SB_DMA_READ(&(data[ptr1].first), 2*sizeof(VTYPE), sizeof(VTYPE), end1-ptr1, P_dot_prod_A);
  SB_DMA_READ(&(data[ptr2].first), 2*sizeof(VTYPE), sizeof(VTYPE), end2-ptr2, P_dot_prod_B);
  SB_DMA_READ(&(data[ptr1].second), 2*sizeof(VTYPE), sizeof(VTYPE), end1-ptr1, P_dot_prod_C);
  SB_DMA_READ(&(data[ptr2].second), 2*sizeof(VTYPE), sizeof(VTYPE), end2-ptr2, P_dot_prod_D);
  SB_CONST(P_dot_prod_A, SENTINAL, 1);
  SB_CONST(P_dot_prod_B, SENTINAL, 1);
  SB_CONST(P_dot_prod_C, 0, 1);
  SB_CONST(P_dot_prod_D, 0, 1);
  SB_DMA_WRITE_SIMP(P_dot_prod_R, 1, &accum);
  SB_WAIT_ALL();
   
  return accum;
}

// this is like both are sparse
float func(float *alpha, VTYPE *y, int start_ptr, int end_ptr){
  float output = 0;
  int n = end_ptr - start_ptr;
  if(n==0) { return 0; }
  // printf("computation of svm function starts with n: %d\n", n);
  SB_CONFIG(svm_func_config, svm_func_size);

  SB_DMA_READ(&gram_mat_pair[start_ptr].second, 2*sizeof(VTYPE), sizeof(VTYPE), n, P_svm_func_gram_mat);
  SB_DMA_READ(&gram_mat_pair[start_ptr].first, 2*sizeof(VTYPE), sizeof(VTYPE), n, P_IND_DOUB0);
  SB_CONFIG_INDIRECT(T64, T64, 1);
  SB_INDIRECT(P_IND_DOUB0, &alpha[0], n, P_svm_func_alpha);
  SB_INDIRECT(P_IND_DOUB1, &y[0], n, P_svm_func_y);
  // SB_CONFIG_INDIRECT(T64, T64, 2);
  // SB_INDIRECT(P_IND_TRIP2, &gram_mat_pair[0].second, n, P_svm_func_gram_mat);
  SB_2D_CONST(P_svm_func_const, 2, n-1, 1, 1, 1);
  // SB_CONST(P_svm_func_b, b, 1);
  SB_DMA_WRITE_SIMP(P_svm_func_output, 1, &output);
  SB_WAIT_ALL();

  // printf("computation of svm function ends\n");
  return output;
}

VTYPE norm(std::pair<VTYPE,VTYPE> *data, int start_ptr, int end_ptr){
  VTYPE output=0;
  int n = end_ptr-start_ptr;
  SB_CONFIG(norm_config, norm_size);
  SB_DMA_READ(&data[start_ptr], 2*sizeof(VTYPE), sizeof(VTYPE), n, P_norm_A);
  SB_2D_CONST(P_norm_const, 2, n-1, 1, 1, 1);
  SB_DMA_WRITE_SIMP(P_norm_output, 1, &output);
  SB_WAIT_ALL();

  return output;
}

void train(std::pair<VTYPE,VTYPE> *data, int *row_ptr, int nnz, VTYPE *y){
  float alpha[M];
  float b1, b2, b=0; // initial bias?
  for(int i=0; i<M; ++i){
    alpha[i]=0.1;
  }
  float E[M]; // let's see the dim of E
  float L = 0, H = 0;
  int passes=0;
  int num_changed_alphas=0;
  float old_alpha_i=0, old_alpha_j=0;
  VTYPE eta = 0;
  float diff = 0;
  int j=0;

  begin_roi();
  mm(data, row_ptr, data, row_ptr, nnz, nnz); // how to access a symmetric sparse matrix- will be a problem?
  // gram_mat_pair, gram_ptr

  while(passes<max_passes){
    num_changed_alphas=0;
    for(int i=0; i<M; ++i){
      E[i] = func(alpha, y, gram_ptr[i], gram_ptr[i+1]); // +b-y[i]
      // printf("Let's print the value of E[i]: %f\n",E[i]);
      if((y[i]*E[i] < -tol && alpha[i]<C) || (y[i]*E[i] > tol && alpha[i]>0)){
        // j = std::rand()%M; // should not be equal to i (random number generation in CGRA?)
        // j = (j + (j+1)%i)%M; // very complicated?
        j = (i+1)%M;
        E[j] = func(alpha, y, gram_ptr[j], gram_ptr[j+1]);
        // printf("Let's print the value of E[j]: %f\n",E[j]);
        old_alpha_i=alpha[i];
        old_alpha_j=alpha[j];
        // eqn(10) and eqn(11)
        if(y[i] != y[j]){
            L = max(0,alpha[j]-alpha[i]);
            H = min(C, C+alpha[j]-alpha[i]);
        } else {
            L = max(0, alpha[i]+alpha[j]-C);
            H = min(C, alpha[i]+alpha[j]);
        }
        if(L==H) continue;
        // eta = eqn(14)
        // eta in dense format??- do something for it
        VTYPE inter_prod = dot_prod_sparse(data, row_ptr[i], row_ptr[i+1], row_ptr[j], row_ptr[j+1]);
        VTYPE intra_prod1 = norm(data, row_ptr[i], row_ptr[i+1]);
        VTYPE intra_prod2 = norm(data, row_ptr[j], row_ptr[j+1]);
        eta = 2*inter_prod - intra_prod1 - intra_prod2;
        // printf("Let's print the value of eta: %lf\n", eta);
        // some condition also:- stream in gram matrix is better
        if(eta >= 0) continue;
        // eqn(12) and eqn(15)
        alpha[j] = alpha[j] - (y[j]*(E[i]-E[j]))/eta; // y should be stored in dense format?
        if(alpha[j]>H){
            alpha[j]=H;
        } else if(alpha[j]<L){
            alpha[j]=L;
        }
        diff = alpha[j]-old_alpha_j;
        if(diff < 1e-5) {
            continue;
        }
        // eqn(16)
        alpha[i]=alpha[i]+ (y[i]*y[j])*(old_alpha_j-alpha[j]);
        // eqn(17) and eqn(18) + eqn(19)
        b1 = b - E[i] - y[i]*diff*intra_prod1 - y[j]*diff*inter_prod;
        b2 = b - E[j] - y[i]*diff*inter_prod - y[j]*diff*intra_prod2;
        if(alpha[i]>0 && alpha[i]<C){
            b = b1;
        } else if(alpha[j]>0 && alpha[j]<C){
            b = b2;
        } else {
            b = (b1+b2)/2;
        }
        num_changed_alphas += 1;
      } 
    }
    if(num_changed_alphas==0){
        passes++;
    } else {
        passes=0;
    }
    // printf("A pass complete\n");
  }
  end_roi();
  sb_stats();
}

int main(){

  int nnz1=0, nrows1=0;
  int *row_ptr1;
  std::pair<VTYPE,VTYPE> *matrix1;

  FILE *m1_file;
  char lineToRead[5000];

   m1_file = fopen("datasets/ddup_pdb1HYS.mtx", "r");

   int id = -1; // pointer to location in values
   int row_id = 0;
   int prev_row_id = -1;
   VTYPE t1 = 0, t2 = 0;
   bool start=false;
   // int x = 0, y = 0;
   printf("Start reading matrix1\n");
   while(fgets(lineToRead, 500000, m1_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &nrows1, &nrows1, &nnz1);
           row_ptr1 = (int*)malloc((nrows1+1)*sizeof(int));
           for(int i=0; i<nrows1; ++i){
               row_ptr1[i] = -1;
           }
           matrix1 = (std::pair<VTYPE,VTYPE>*)malloc(nnz1*sizeof(std::pair<VTYPE,VTYPE>));
           start = true;
       }

       else {
            ++id;
            // sscanf(lineToRead, "%ld %d %ld", &t1, &row_id, &t2);
            sscanf(lineToRead, "%lf %d %lf", &t1, &row_id, &t2);
            row_id--;
            matrix1[id] = std::make_pair((t1-1), t2);
            if(row_ptr1[row_id]==-1) {
                row_ptr1[row_id] = id;
                if(prev_row_id!=-1){
                  for(int i=prev_row_id+1; i<row_id; ++i){
                      // assert(row_ptr1[i]==-1 && "some problem\n");
                      row_ptr1[i] = id;
                  }
                }
                prev_row_id = row_id;
            }
       }
    }
    row_ptr1[nrows1] = nnz1-1;

    VTYPE *y;
    srand(1); // this is seed is to be used by the algorithm
    y = (VTYPE*)malloc(M*sizeof(VTYPE));
    for(int i=0; i<M; ++i){
        y[i] = rand()%10;
    }

   printf("Finished reading input data\n");

  train(matrix1, row_ptr1, nnz1, y);
  printf("svm training done\n");
  
  return 0;
}
