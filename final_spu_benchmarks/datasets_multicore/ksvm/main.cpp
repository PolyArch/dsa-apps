#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <inttypes.h>
#include <vector>
#include <math.h>
#include "ksvm.dfg.h"
#include "duality_gap.dfg.h"
#include "eta.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"

using namespace std;

#define float float
#define C 0.1
#define tol 0.02
// #define max_passes 10
#define max_passes 1
// #define M 52 // number of instances
// #define N 52 // number of features
#define sigma 0.5

#define INT16MAX ((1<<16)-1)

// very small.data
// #define N 84
// #define M 10

// #define N 84
// #define M 100

std::pair<float,float> *gram_mat_pair;
int *gram_ptr;

// input train set
uint64_t y[M];
vector<uint32_t> data_val[M];
vector<uint32_t> data_ind[M];
// int row_ptr[M+1];

float min(float a, float b){
  return a<b?a:b;
}

float max(float a, float b){
  return a>b?a:b;
}

void eta_calc(int i, int j, double &dp, double &norm1, double &norm2){
  
  if(data_val[i].size()==0 || data_val[j].size()==0)
	return;
  cout << data_val[i].size() << " " << data_val[j].size() << "\n";
  
  SB_CONFIG(eta_config, eta_size);

  // SB_DMA_READ(&data_ind[i][0], sizeof(float), sizeof(float), data_ind[i].size(), P_eta_a_ind);
  // SB_DMA_READ(&data_val[i][0], sizeof(float), sizeof(float), data_val[i].size(), P_eta_a_val);
  // SB_DMA_READ(&data_ind[j][0], sizeof(float), sizeof(float), data_ind[j].size(), P_eta_b_ind);
  // SB_DMA_READ(&data_val[j][0], sizeof(float), sizeof(float), data_val[j].size(), P_eta_b_val);

  // TODO:FIXME: see how to remove padding for ksvm 
  SB_DMA_READ(&data_ind[i][0], 8, 8, data_ind[i].size()/2, P_eta_a_ind);
  SB_DMA_READ(&data_val[i][0], 8, 8, data_val[i].size()/2, P_eta_a_val);
  SB_DMA_READ(&data_ind[j][0], 8, 8, data_ind[j].size()/2, P_eta_b_ind);
  SB_DMA_READ(&data_val[j][0], 8, 8, data_val[j].size()/2, P_eta_b_val);
 
  // 32-bit sentinal? (how does 64-bit works -- some trick in IM32x2)
  SB_CONST(P_eta_a_ind, SENTINAL, 1);
  SB_CONST(P_eta_b_ind, SENTINAL, 1);
  SB_CONST(P_eta_a_val, 0, 1);
  SB_CONST(P_eta_b_val, 0, 1);
  SB_2D_CONST(P_eta_const1, 2, (data_ind[i].size())/2-1, 1, 1, 1);
  SB_2D_CONST(P_eta_const2, 2, (data_ind[i].size())/2-1, 1, 1, 1);
  // double norm1, norm2, dp;
  SB_DMA_WRITE_SIMP(P_eta_n1, 1, &norm1);
  SB_DMA_WRITE_SIMP(P_eta_n2, 1, &norm2);
  SB_DMA_WRITE_SIMP(P_eta_s, 1, &dp);
  SB_WAIT_ALL();

  cout << "Eta calc done\n";
}

void calc_duality_gap(double alpha[M], double E[M], double b, double &duality_gap){
  SB_CONFIG(duality_gap_config, duality_gap_size);

  SB_DMA_READ(&alpha[0], 8, 8, M, P_duality_gap_alpha);
  SB_DMA_READ(&y[0], 8, 8, M, P_duality_gap_y);
  SB_DMA_READ(&E[0], 8, 8, M, P_duality_gap_E);
  SB_CONST(P_duality_gap_b, b, 1);
  SB_2D_CONST(P_duality_gap_const, 2, M-1, 1, 1, 1);

  SB_DMA_WRITE_SIMP(P_duality_gap_dgap, 1, &duality_gap);
  SB_WAIT_ALL();
  cout << "Duality calc done\n";
}


void kernel_err_update(int i, int j, double diff1, double diff2, double y1, double y2, double (&E)[M]){
  
  if(data_val[i].size()==0 || data_val[j].size()==0)  return;

  // double output = 0;
  int num_inst = M;
  double gauss_var = -1/(2*sigma*sigma); // double to fix
  // int m=1;
  SB_CONFIG(ksvm_config, ksvm_size);

  SB_CONST(P_ksvm_gauss_var, DOUBLE_TO_FIX(gauss_var), num_inst);
  SB_CONST(P_ksvm_alpha1, diff1, num_inst);
  SB_CONST(P_ksvm_alpha2, diff2, num_inst);
  SB_CONST(P_ksvm_y1, y1, num_inst);
  SB_CONST(P_ksvm_y2, y2, num_inst);
  SB_DMA_READ(&E[0], 8, 8, num_inst, P_ksvm_old_E);

  // SB_2D_CONST(P_ksvm_const, 2, num_inst-1, 1, 1, 1);
  // SB_2D_CONST(P_ksvm_const, 2, 0, 1, 1, num_inst);
  // SB_CONST(P_ksvm_const, 1, num_inst);
  for(int k=0; k<num_inst; ++k){
    // std::cout << "k: " << k << " a_count: " << (end3-ptr3)/2 << " b_count: " << (end2-ptr2)/2 << " c_count: :" << (end1-ptr1)/2 << "\n";
	
	if(data_val[k].size()==0)
	  continue;
    SB_DMA_READ(&data_ind[k][0], sizeof(float), sizeof(float), data_ind[k].size(), P_ksvm_a_ind);
    SB_DMA_READ(&data_val[k][0], sizeof(float), sizeof(float), data_val[k].size(), P_ksvm_a_val);
    SB_DMA_READ(&data_ind[i][0], sizeof(float), sizeof(float), data_ind[i].size(), P_ksvm_b_ind);
    SB_DMA_READ(&data_val[i][0], sizeof(float), sizeof(float), data_val[i].size(), P_ksvm_b_val);
    SB_DMA_READ(&data_ind[j][0], sizeof(float), sizeof(float), data_ind[j].size(), P_ksvm_c_ind);
    SB_DMA_READ(&data_val[j][0], sizeof(float), sizeof(float), data_val[j].size(), P_ksvm_c_val);

    // 32-bit sentinal?
    SB_CONST(P_ksvm_a_ind, SENTINAL, 1);
    SB_CONST(P_ksvm_b_ind, SENTINAL, 1);
    SB_CONST(P_ksvm_c_ind, SENTINAL, 1);
    SB_CONST(P_ksvm_a_val, 0, 1);
    SB_CONST(P_ksvm_b_val, 0, 1);
    SB_CONST(P_ksvm_c_val, 0, 1);
  }
  // SB_DMA_WRITE_SIMP(P_err_error, 1, &output);
  // SB_DMA_WRITE_SIMP(P_ksvm_E, num_inst-9, &E[0]);
  SB_DMA_WRITE_SIMP(P_ksvm_E, num_inst, &E[0]);
  SB_WAIT_ALL();
  cout << "Kernel err calc done\n";
}

void train(){
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
  // int num_changed_alphas=0;
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
	cout << L << " " << H << endl;
    if(L==H) continue;
    double inter_prod = 0, norm1 = 0, norm2 = 0;
	cout << "Sent for eta calculation\n";
    eta_calc(i, j, inter_prod, norm1, norm2);
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

	cout << "Sent for kernel err calculation\n";
    kernel_err_update(i, j, diff1, diff, y[i], y[j], E);

    duality_gap = 0;
	cout << "Sent for duality gap calculation\n";
    calc_duality_gap(alpha, E, b, duality_gap);
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


  FILE *m1_file;
  char lineToRead[5000];
  string str(file);

  m1_file = fopen(str.c_str(), "r");

  // m1_file = fopen("datasets/small_adult.data", "r");
  // m1_file = fopen("datasets/very_small.data", "r");
  printf("Start reading matrix1\n");

  int inst_id=0;
  while(fgets(lineToRead, 5000, m1_file) != NULL) {
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	char ignore;
	float x;
	int ind;

	iss >> x;
	y[inst_id] = DOUBLE_TO_FIX(x);

	while(iss >> ind) {
	  iss >> ignore >> x;
	  data_ind[inst_id].push_back(ind);
	  data_val[inst_id].push_back(DOUBLE_TO_FIX(x));
	}
	
    inst_id++;;
  }

  printf("Finished reading input data\n");

  train();
  printf("svm training done\n");
 
  return 0;
}
