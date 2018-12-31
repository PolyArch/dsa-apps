#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <inttypes.h>
#include <math.h>
#include "new_err.dfg.h"
#include "duality_gap.dfg.h"
#include "eta.dfg.h"
#include "../../common/include/ss_insts.h"
#include "../../common/include/sim_timing.h"
using namespace std;

// #define VTYPE double
#define VTYPE float
#define C 0.1
#define tol 0.02
// #define max_passes 10
#define max_passes 1
// #define M 52 // number of instances
// #define N 52 // number of features
#define sigma 0.5
 
#ifndef FxPnt
#define FxPnt 8
#endif

#define INT16MAX ((1<<16)-1)

#define fx_to_flt(x) ((float)(x) / (1<<FxPnt))
#define flt_to_fx(x) ((int)((x) * (1<<FxPnt)))

 
std::pair<VTYPE,VTYPE> *gram_mat_pair;
int *gram_ptr;

float min(float a, float b){
  return a<b?a:b;
}
float max(float a, float b){
  return a>b?a:b;
}

void eta_calc(uint32_t *data_val, uint32_t *data_ind, int ptr1, int end1, int ptr2, int end2, double &dp, double &norm1, double &norm2){
  // std::cout << "Size1: " << (end1-ptr1) << " and size2: " << (end2-ptr2) << endl;
  if(end1==ptr1 || end2==ptr2)
    return;
  SS_CONFIG(eta_config, eta_size);
  SS_DMA_READ(&data_ind[ptr1], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end1-ptr1)/2, P_eta_a_ind);
  SS_DMA_READ(&data_val[ptr1], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end1-ptr1)/2, P_eta_a_val);
  SS_DMA_READ(&data_ind[ptr2], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end2-ptr2)/2, P_eta_b_ind);
  SS_DMA_READ(&data_val[ptr2], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end2-ptr2)/2, P_eta_b_val);
  // 32-bit sentinal?
  SS_CONST(P_eta_a_ind, SENTINAL, 1);
  SS_CONST(P_eta_b_ind, SENTINAL, 1);
  SS_CONST(P_eta_a_val, 0, 1);
  SS_CONST(P_eta_b_val, 0, 1);
  SS_2D_CONST(P_eta_const1, 2, (end1-ptr1)/2-1, 1, 1, 1);
  SS_2D_CONST(P_eta_const2, 2, (end2-ptr2)/2-1, 1, 1, 1);
  // double norm1, norm2, dp;
  SS_DMA_WRITE_SIMP(P_eta_n1, 1, &norm1);
  SS_DMA_WRITE_SIMP(P_eta_n2, 1, &norm2);
  SS_DMA_WRITE_SIMP(P_eta_s, 1, &dp);
  SS_WAIT_ALL();

  // return (2*dp - norm1 - norm2);
}

void calc_duality_gap(double alpha[M], uint64_t y[M], double E[M], double b, double &duality_gap){
  SS_CONFIG(duality_gap_config, duality_gap_size);

  SS_DMA_READ(&alpha[0], 8, 8, M, P_duality_gap_alpha);
  SS_DMA_READ(&y[0], 8, 8, M, P_duality_gap_y);
  SS_DMA_READ(&E[0], 8, 8, M, P_duality_gap_E);
  SS_CONST(P_duality_gap_b, b, 1);
  SS_2D_CONST(P_duality_gap_const, 2, M-1, 1, 1, 1);

  SS_STRIDE(8,8);
  SS_DMA_WRITE_SIMP(P_duality_gap_dgap, 1, &duality_gap);
  SS_WAIT_ALL();
}


void kernel_err_update(uint32_t *data_val, uint32_t *data_ind, int *row_ptr, int i, int j, double diff1, double diff2, double y1, double y2, double (&E)[M]){
  // double output = 0;
  int num_inst = M;
  // num_inst=20;
  int ptr1 = row_ptr[i];
  int end1 = row_ptr[i+1];
  // need to padding
  // ptr1 = 0; end1 = 20;
  int ptr2, end2;
  int ptr3, end3;
  ptr2 = row_ptr[j];
  end2 = row_ptr[j+1];
  if(end1==ptr1 || end2==ptr2)
    return;

  // diff1 = 2;
  // diff2 = 3;
  // should copy uint64_t?

  double gauss_var = -1/(2*sigma*sigma); // double to fix
  // int m=1;
  SS_CONFIG(new_err_config, new_err_size);

  SS_CONST(P_new_err_gauss_var, gauss_var, num_inst);
  SS_CONST(P_new_err_alpha1, diff1, num_inst);
  SS_CONST(P_new_err_alpha2, diff2, num_inst);
  SS_CONST(P_new_err_y1, y1, num_inst);
  SS_CONST(P_new_err_y2, y2, num_inst);
  SS_DMA_READ(&E[0], 8, 8, num_inst, P_new_err_old_E);

  // SS_2D_CONST(P_new_err_const, 2, num_inst-1, 1, 1, 1);
  // SS_2D_CONST(P_new_err_const, 2, 0, 1, 1, num_inst);
  // SS_CONST(P_new_err_const, 1, num_inst);
  for(int k=0; k<num_inst; ++k){
    ptr3 = row_ptr[k];
    end3 = row_ptr[k+1];
    // std::cout << "k: " << k << " a_count: " << (end3-ptr3)/2 << " b_count: " << (end2-ptr2)/2 << " c_count: :" << (end1-ptr1)/2 << "\n";
    // ptr2 = 0; end2 = 20;
    SS_DMA_READ(&data_ind[ptr3], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end3-ptr3)/2, P_new_err_a_ind);
    SS_DMA_READ(&data_val[ptr3], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end3-ptr3)/2, P_new_err_a_val);
    SS_DMA_READ(&data_ind[ptr1], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end1-ptr1)/2, P_new_err_b_ind);
    SS_DMA_READ(&data_val[ptr1], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end1-ptr1)/2, P_new_err_b_val);
    SS_DMA_READ(&data_ind[ptr2], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end2-ptr2)/2, P_new_err_c_ind);
    SS_DMA_READ(&data_val[ptr2], 2*sizeof(VTYPE), 2*sizeof(VTYPE), (end2-ptr2)/2, P_new_err_c_val);

    // 32-bit sentinal?
    SS_CONST(P_new_err_a_ind, SENTINAL, 1);
    SS_CONST(P_new_err_b_ind, SENTINAL, 1);
    SS_CONST(P_new_err_c_ind, SENTINAL, 1);
    SS_CONST(P_new_err_a_val, 0, 1);
    SS_CONST(P_new_err_b_val, 0, 1);
    SS_CONST(P_new_err_c_val, 0, 1);
  }
  // SS_DMA_WRITE_SIMP(P_err_error, 1, &output);
  // SS_DMA_WRITE_SIMP(P_new_err_E, num_inst-9, &E[0]);
  SS_DMA_WRITE_SIMP(P_new_err_E, num_inst, &E[0]);
  SS_WAIT_ALL();
  // return output+b-y[i];
}





// void train(std::pair<VTYPE,VTYPE> *data, int *row_ptr, int nnz, VTYPE *y){
void train(uint32_t* data_val, uint32_t* data_ind, int *row_ptr, uint64_t *y){
  // float alpha[M];
  double alpha[M];
  // float b1, b2, b=0; // initial bias?
  double b1, b2, b=0; // initial bias?
  for(int i=0; i<M; ++i){
    alpha[i]=0;
    // alpha[i]=0.1;
  }
  // float E[M]; // let's see the dim of E
  double E[M]; // let's see the dim of E
  for(int k=0; k<M; ++k){
    E[k] = -y[k];
  }

  // float L = 0, H = 0;
  double L = 0, H = 0;
  int passes=0;
  int num_changed_alphas=0;
  // float old_alpha_i=0, old_alpha_j=0;
  double old_alpha_i=0, old_alpha_j=0;
  double eta = 0;
  // float diff = 0;
  double diff = 0;
  int j=1, i=0;
  double duality_gap=0;
  double dual=0;

  begin_roi();

  // while (duality_gap <= tol*dual && passes<max_passes) {
  // while (duality_gap <= tol*dual || passes<max_passes) {
  while (passes<max_passes) {
    passes++;

    // cout << "Pass number: " << passes << "\n";
    // Select new i and j such that E[i] is max and E[j] is min
    // do in CGRA
    for(int k=0; k<M; ++k){
      if(E[k]>E[i])
        i=k;
      if(E[k]<E[j])
        j=k;
    }
	std::cout << "i: " << i << " j: " << j << std::endl;
	// j = rand()%M;
	// j = 2;
	// j = 20;

    // Step 1:
    old_alpha_i=alpha[i];
    old_alpha_j=alpha[j];
    if(y[i] != y[j]){
      L = max(0,alpha[j]-alpha[i]);
      H = min(C, C+alpha[j]-alpha[i]);
    } else {
      L = max(0, alpha[i]+alpha[j]-C);
      H = min(C, alpha[i]+alpha[j]);
    }
    // cout << "L=H?\n";
    if(L==H) continue;
    // VTYPE inter_prod = dot_prod_sparse(data, row_ptr[i], row_ptr[i+1], row_ptr[j], row_ptr[j+1]);
    // VTYPE intra_prod1 = norm(data, row_ptr[i], row_ptr[i+1]);
    // VTYPE intra_prod2 = norm(data, row_ptr[j], row_ptr[j+1]);
    // eta = 2*inter_prod - intra_prod1 - intra_prod2;
    double inter_prod = 0, norm1 = 0, norm2 = 0;
    eta_calc(data_val, data_ind, row_ptr[i], row_ptr[i+1], row_ptr[j], row_ptr[j+1], inter_prod, norm1, norm2);
    eta = 2*inter_prod - norm1 - norm2;
    if(eta == 0) eta=2;
    // cout << "Eta was less\n";
    // if(eta >= 0) continue;
    // double diff2 = (y[j]*(E[i]-E[j]))/eta;
    diff = (y[j]*(E[i]-E[j]))/eta;
    alpha[j] = alpha[j] - diff;
    // alpha[j] = alpha[j] - diff2;
    // alpha[j] = alpha[j] - (y[j]*(E[i]-E[j]))/eta; // y should be stored in dense format?
    if(alpha[j]>H){
        alpha[j]=H;
    } else if(alpha[j]<L){
        alpha[j]=L;
    }
    // diff = alpha[j]-old_alpha_j;
    /*
    cout << "Diff was less\n";
    if(diff < 1e-5) {
        continue;
    }
    */
    // double diff1 = (y[i]*y[j])*(diff2);
    double diff1 = (y[i]*y[j])*(diff);
    // alpha[i]=alpha[i]+ (y[i]*y[j])*(old_alpha_j-alpha[j]);
    alpha[i]=alpha[i]+ diff1;

    // b1 = b - E[i] - y[i]*diff;
    // b2 = b - E[j] - y[i]*diff;

    b1 = b - E[i] - y[i]*diff*norm1 - y[j]*diff*inter_prod;
    b2 = b - E[j] - y[i]*diff*inter_prod - y[j]*diff*norm2;

    if(alpha[i]>0 && alpha[i]<C){
        b = b1;
    } else if(alpha[j]>0 && alpha[j]<C){
        b = b2;
    } else {
        b = (b1+b2)/2;
    }
    dual = dual - diff/y[i]*(E[i]-E[j]) + eta/2*(diff/y[i])*(diff/y[i]);

    // kernel_err_update(data_val, data_ind, row_ptr, i, j, diff1, diff2, y[i], y[j], E);
    kernel_err_update(data_val, data_ind, row_ptr, i, j, diff1, diff, y[i], y[j], E);

    duality_gap = 0;
    calc_duality_gap(alpha, y, E, b, duality_gap);
    /*
    for(int k=0; k<M; ++k){
      duality_gap += alpha[k]*y[k]*E[k];
    }
    duality_gap += b;
    */

  }

  end_roi();
  sb_stats();
}

int main(){

  int nnz1=0, nrows1=0;
  int *row_ptr;
  // std::pair<VTYPE,VTYPE> *matrix1;
  uint32_t *data_val;
  uint32_t *data_ind;

  uint64_t *y;
  y = (uint64_t*)malloc(M*sizeof(uint64_t));
  data_val = (uint32_t*)malloc(M*N*ratio*sizeof(uint32_t));
  data_ind = (uint32_t*)malloc(M*N*ratio*sizeof(uint32_t));
  row_ptr = (int*)malloc((M+1)*sizeof(int));


  float *temp_val;
  float *temp_ind;
  double *out;
  out = (double*)malloc(1*sizeof(double));
  temp_val = (float*)malloc(N*ratio*sizeof(float));
  temp_ind = (float*)malloc(N*ratio*sizeof(float));

  int id=0;


  FILE *m1_file;
  char lineToRead[5000];

  m1_file = fopen("input.data", "r");
  printf("Start reading matrix1\n");

  while(fgets(lineToRead, 500000, m1_file) != NULL) {
	// I need to read N*ratio values
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	char ignore;

	iss >> out[0];

	for(int i=0; i<N*ratio; ++i){
	  iss >> ignore >> temp_ind[i] >> ignore >> temp_val[i];
	}
	
    // sscanf(lineToRead, "%lf %f:%f %f:%f %f:%f %f:%f %f:%f %f:%f", &out[0], &temp_ind[0], &temp_val[0], &temp_ind[1], &temp_val[1], &temp_ind[2], &temp_val[2], &temp_ind[3], &temp_val[3], &temp_ind[4], &temp_val[4], &temp_ind[5], &temp_val[6]);
    y[id] = (int32_t)(out[0] * (1<<FxPnt));
    for(int j=0; j<N*ratio; ++j){
      // data_val[id*N*ratio + j] = fx_to_flt(temp_val[j]);
      data_val[(int)(id*N*ratio + j)] = (int32_t)(temp_val[j] * (1<<FxPnt));
      data_ind[(int)(id*N*ratio + j)] = (int32_t)(temp_ind[j] * (1<<FxPnt));
      // data_ind[id*N*ratio + j] = fx_to_flt(temp_ind[j]);
    }
    row_ptr[id] = id*N*ratio;
    id++;
  }
  row_ptr[M] = N*M*ratio;

  printf("Finished reading input data\n");

  train(data_val, data_ind, row_ptr, y);
  printf("svm training done\n");
 
  return 0;
}
