#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <inttypes.h>
#include <assert.h>
#include <vector>
#include <math.h>
#include "ksvm.dfg.h"
#include "duality_gap.dfg.h"
#include "eta.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"

using namespace std;

#define float float
#define C 0.1
#define tol 0.02
// #define max_passes 10
#define max_passes 1
#define sigma 0.5

#define NUM_THREADS	64
#define INT16MAX ((1<<16)-1)

// input train set
uint64_t y[M];
uint32_t data[M][N];
  
double alpha[M];
double E[M];

// Barrier variable
pthread_barrier_t barr;

float min(float a, float b){
  return a<b?a:b;
}

float max(float a, float b){
  return a>b?a:b;
}

// done at only 1 core
void eta_calc(long tid, int i, int j, double &dp, double &norm1, double &norm2){

  SB_CONFIG(eta_config, eta_size);
  
  // broadcast_eta(tid, i,j);
  SB_DMA_READ(&data[i][0], 8, 8, N/2, P_eta_a_val);
  SB_DMA_READ(&data[j][0], 8, 8, N/2, P_eta_b_val);

  SB_CONST(P_eta_IM, 2, N/2-1); // acc, discard
  SB_CONST(P_eta_IM, 0, 1); // no disc

  // FIXME
  SB_2D_CONST(P_eta_const1, 2, N/2-1, 1, 1, 1);
  SB_2D_CONST(P_eta_const2, 2, N/2-1, 1, 1, 1);
  
  SB_DMA_WRITE_SIMP(P_eta_n1, 1, &norm1);
  SB_DMA_WRITE_SIMP(P_eta_n2, 1, &norm2);
  SB_DMA_WRITE_SIMP(P_eta_s, 1, &dp);
  SB_WAIT_ALL();

  // cout << "Eta calc done\n";
}

void calc_duality_gap(long tid, double b, double &duality_gap){
  SB_CONFIG(duality_gap_config, duality_gap_size);

  int num_inst = M/NUM_THREADS;

  int start_id = tid*num_inst;
  int end_id = start_id+num_inst;

  SB_SCRATCH_READ(getBankedOffset(2,3), 8*num_inst, P_duality_gap_alpha);
  SB_SCRATCH_READ(getBankedOffset(1,3), 8*num_inst, P_duality_gap_y);
  SB_SCRATCH_READ(getBankedOffset(0,3), 8*num_inst, P_duality_gap_E);
  
  SB_CONST(P_duality_gap_b, b, 1);
  // SB_2D_CONST(P_duality_gap_const, 2, num_inst-1, 0, 1, 1);
  SB_2D_CONST(P_duality_gap_const, 2, num_inst-1, 1, 1, 1);

  SB_STRIDE(8,8);
  SB_DMA_WRITE_SIMP(P_duality_gap_dgap, 1, &duality_gap);
  SB_WAIT_ALL();
  // cout << "Duality calc done\n";
}

void kernel_err_update(long tid, int i, int j, double diff1, double diff2, double y1, double y2){
  
  // int num_inst = M;
  int num_inst = M/NUM_THREADS; // 46
  double gauss_var = -1/(2*sigma*sigma); // double to fix
  
  SB_CONFIG(ksvm_config, ksvm_size);

  SB_CONST(P_ksvm_gauss_var, DOUBLE_TO_FIX(gauss_var), num_inst);
  SB_CONST(P_ksvm_alpha1, diff1, num_inst);
  SB_CONST(P_ksvm_alpha2, diff2, num_inst);
  SB_CONST(P_ksvm_y1, y1, num_inst);
  SB_CONST(P_ksvm_y2, y2, num_inst);

  // banked scratch read
  SB_SCRATCH_READ(getBankedOffset(0,3), 8*num_inst, P_ksvm_old_E);

  int start_id = tid*num_inst;
  int end_id = start_id+num_inst;

  // this is exceeding address space
  // SB_SCRATCH_READ(getLinearAddr(0), num_inst*4*N ,P_ksvm_a_val);
  SB_DMA_READ(&data[start_id][0], 8, 8, N/2*num_inst, P_ksvm_a_val);
  
  // SB_2D_CONST(P_eta_IM, 2, N/2-1, 0, 1, num_inst);
  SB_2D_CONST(P_eta_IM, 2, N/2-1, 1, 1, num_inst);

  // broadcast_inst(tid, i, j);
  for(int k=0; k<num_inst; ++k) {
    SB_SCRATCH_READ(getLinearAddr(0), N*4, P_ksvm_b_val);
    SB_SCRATCH_READ(getLinearAddr(getLinearOffset(1,2)), N*4, P_ksvm_c_val);
 
    // SB_DMA_READ(&data[i][0], 8, 8, N/2, P_ksvm_b_val);
    // SB_DMA_READ(&data[j][0], 8, 8, N/2, P_ksvm_c_val);
  }

  // write this in banked scratch
  SB_SCR_WRITE(P_ksvm_E, num_inst*8, getBankedOffset(0,3));
  SB_WAIT_ALL();
  // cout << "Kernel err calc done\n";
}

void train(long tid) {
  double b1, b2, b=0; // initial bias?

  double L = 0, H = 0;
  int passes=0;
  double old_alpha_i=0, old_alpha_j=0;
  double eta = 0;
  // float diff = 0;
  double diff = 0;
  int j=1, i=0;
  double duality_gap=0;
  double dual=0;

  while (passes<max_passes) {
    passes++;
    double inter_prod = 0, norm1 = 0, norm2 = 0;

    // Select new i and j such that E[i] is max and E[j] is min do in CGRA
    for(int k=0; k<M; ++k){
      if(E[k]>E[i])
        i=k;
      if(E[k]<E[j])
        j=k;
    }
	// std::cout << "i: " << i << " j: " << j << std::endl;

    begin_roi();
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
    
    // begin_roi();
    eta_calc(tid, i, j, inter_prod, norm1, norm2);
    // end_roi();
    // sb_stats();

    eta = 2*inter_prod - norm1 - norm2;
    if(eta == 0) eta=2;
    diff = (y[j]*(E[i]-E[j]))/eta;
    alpha[j] = alpha[j] - diff;
    if(alpha[j]>H){
        alpha[j]=H;
    } else if(alpha[j]<L){
        alpha[j]=L;
    }
    
    double diff1 = (y[i]*y[j])*(diff);
    alpha[i]=alpha[i]+ diff1;

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

    // begin_roi();
    kernel_err_update(tid, i, j, diff1, diff, y[i], y[j]);
    // end_roi();
    // sb_stats();

    duality_gap = 0;
    // begin_roi();
    calc_duality_gap(tid, b, duality_gap);
    end_roi();
    sb_stats();
    
  }
}


void load_linear_scratch(long tid) {

  int val_offset = getLinearAddr(getLinearOffset(0,2));

  int n_inst = M/NUM_THREADS;
  for(int i=tid*n_inst; i<(tid+1)*n_inst; ++i) {
    SB_DMA_SCRATCH_LOAD(&data[i][0], 4, 4, N, val_offset);
    val_offset += M*4;
  }
  SB_WAIT_SCR_WR();
}

void *entry_point(void *threadid) {
  
   long tid;
   tid = (long)threadid;
   // Synchronization point
   int rc = pthread_barrier_wait(&barr);
   if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
   {
     printf("Could not wait on barrier\n");
     // exit(-1);
   }

   SB_CONFIG(eta_config, eta_size);
   load_linear_scratch(tid);
   begin_roi();
   train(tid);
   end_roi();
   sb_stats();
   // pthread_exit(NULL);
   return NULL;
}

int main(){

  for(int i=0; i<M; ++i){
    alpha[i]=0;
    // alpha[i]=0.1;
  }
 
  for(int k=0; k<M; ++k){
    E[k] = -y[k];
  }

  // m1_file = fopen("datasets/small_adult.data", "r");
  // m1_file = fopen("datasets/very_small.data", "r");
  printf("Start reading matrix1\n");

  /*
   FILE *m1_file;
  char lineToRead[5000];
  string str(file);

  m1_file = fopen(str.c_str(), "r");


  int inst_id=0;
  int cur_v=0;
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
	  data[inst_id][cur_v] = DOUBLE_TO_FIX(x);
      cur_v++;
	}
    cur_v=0;
    
    inst_id++;
  }
  */

  // for(int i=0; i<M; ++i) {
  for(int i=0; i<M/64; ++i) {
    for(int j=0; j<N; ++j) {
      data[i][j] = rand()%50;
    }
    y[i]=1;
  }

  printf("Finished reading input data\n");


  // begin_roi();
  train(0);
  // end_roi();
  // sb_stats();

  printf("svm training done\n");
 
  return 0;
}
