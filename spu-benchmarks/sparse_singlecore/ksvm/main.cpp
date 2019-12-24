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
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
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
vector<uint32_t> data_val[M];
vector<uint32_t> data_ind[M];
int data_ptr[M+1]; // save the accumulated indices
  
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

int getCoreLoc(int inst_id) {
  return NUM_THREADS*inst_id/M;
}

void broadcast_inst(long tid, int i, int j) {
  uint64_t mask=0;
  for(int k=0; k<NUM_THREADS; ++k){
    addDest(mask,k);
  }
  if(tid==getCoreLoc(i)) {
    int local_offset = (data_ptr[i]-data_ptr[tid*M/NUM_THREADS])*4;
    assert(data_ind[i].size() == (data_ptr[i+1]-data_ptr[i]));
    SS_SCR_REM_PORT(getLinearAddr(local_offset), (data_ptr[i+1]-data_ptr[i])*4, mask, P_ksvm_b_val);
    SS_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)+local_offset), (data_ptr[i+1]-data_ptr[i])*4, mask, P_ksvm_b_ind);
  }

  if(tid==getCoreLoc(j)) {
    int local_offset = (data_ptr[j]-data_ptr[tid*M/NUM_THREADS])*4;
    assert(data_ind[j].size() == (data_ptr[j+1]-data_ptr[j]));
    SS_SCR_REM_PORT(getLinearAddr(local_offset), (data_ptr[j+1]-data_ptr[j])*4, mask, P_ksvm_c_val);
    SS_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)+local_offset), (data_ptr[j+1]-data_ptr[j])*4, mask, P_ksvm_c_ind);
  }
}

// FIXME: should not need two same functions?
void broadcast_eta(long tid, int i, int j) {
  uint64_t mask=0;
  addDest(mask,0);
  
  if(tid==getCoreLoc(i)) {
    int local_offset = (data_ptr[i]-data_ptr[tid*M/NUM_THREADS])*4;
    assert(data_ind[i].size() == (data_ptr[i+1]-data_ptr[i]));
    SS_SCR_REM_PORT(getLinearAddr(local_offset), (data_ptr[i+1]-data_ptr[i])*4, mask, P_eta_a_val);
    SS_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)+local_offset), (data_ptr[i+1]-data_ptr[i])*4, mask, P_eta_a_ind);
  }

  if(tid==getCoreLoc(j)) {
    int local_offset = (data_ptr[j]-data_ptr[tid*M/NUM_THREADS])*4;
    assert(data_ind[j].size() == (data_ptr[j+1]-data_ptr[j]));
    SS_SCR_REM_PORT(getLinearAddr(local_offset), (data_ptr[j+1]-data_ptr[j])*4, mask, P_eta_b_val);
    SS_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)+local_offset), (data_ptr[j+1]-data_ptr[j])*4, mask, P_eta_b_ind);
  }
}

void load_ij_linear_scratch(int i, int j) {

  int val1_offset = getLinearAddr(getLinearOffset(0,4));
  int ind1_offset = getLinearAddr(getLinearOffset(1,4));
  int val2_offset = getLinearAddr(getLinearOffset(2,4));
  int ind2_offset = getLinearAddr(getLinearOffset(3,4));

  SS_DMA_SCRATCH_LOAD(&data_val[i][0], 4, 4, data_val[i].size(), val1_offset);
  SS_DMA_SCRATCH_LOAD(&data_ind[i][0], 4, 4, data_ind[i].size(), ind1_offset);

  SS_DMA_SCRATCH_LOAD(&data_val[j][0], 4, 4, data_val[j].size(), val2_offset);
  SS_DMA_SCRATCH_LOAD(&data_ind[j][0], 4, 4, data_ind[j].size(), ind2_offset);
  SS_WAIT_SCR_WR();
  SS_WAIT_ALL();
}


// done at only 1 core
void eta_calc(long tid, int i, int j, double &dp, double &norm1, double &norm2){
  
  if(data_val[i].size()==0 || data_val[j].size()==0)
	return;
 
  int val1_offset = getLinearAddr(getLinearOffset(0,4));
  int ind1_offset = getLinearAddr(getLinearOffset(1,4));
  int val2_offset = getLinearAddr(getLinearOffset(2,4));
  int ind2_offset = getLinearAddr(getLinearOffset(3,4));

 // cout << data_val[i].size() << " " << data_val[j].size() << "\n";
  
  // broadcast_eta(tid, i,j);
  SS_SCRATCH_READ(val1_offset, 8*(data_val[i].size()/2), P_eta_a_val);
  SS_SCRATCH_READ(ind1_offset, 8*(data_ind[i].size()/2), P_eta_a_ind);
  SS_SCRATCH_READ(val2_offset, 8*(data_val[j].size()/2), P_eta_b_val);
  SS_SCRATCH_READ(ind2_offset, 8*(data_ind[j].size()/2), P_eta_b_ind);
 
  // SS_DMA_READ(&data_val[i][0], 8, 8, (data_ptr[i+1]-data_ptr[i])/2, P_eta_a_val);
  // SS_DMA_READ(&data_ind[i][0], 8, 8, (data_ptr[i+1]-data_ptr[i])/2, P_eta_a_ind);
  // SS_DMA_READ(&data_val[j][0], 8, 8, (data_ptr[j+1]-data_ptr[j])/2, P_eta_b_val);
  // SS_DMA_READ(&data_ind[j][0], 8, 8, (data_ptr[j+1]-data_ptr[j])/2, P_eta_b_ind);

  SS_2D_CONST(P_eta_const1, 2, (data_ind[i].size())/2-1, 1, 1, 1);
  SS_2D_CONST(P_eta_const2, 2, (data_ind[i].size())/2-1, 1, 1, 1);
  
  SS_DMA_WRITE(P_eta_n1, 8, 8, 1, &norm1);
  SS_DMA_WRITE(P_eta_n2, 8, 8, 1, &norm2);
  SS_DMA_WRITE(P_eta_s, 8, 8, 1, &dp);
  SS_WAIT_ALL();

  // cout << "Eta calc done\n";
}

void calc_duality_gap(long tid, double b, double &duality_gap){
  SS_CONFIG(duality_gap_config, duality_gap_size);

  int num_inst = M/NUM_THREADS;
  if(num_inst==0) return;

  SS_SCRATCH_READ(getBankedOffset(2,3), 8*num_inst, P_duality_gap_alpha);
  SS_SCRATCH_READ(getBankedOffset(1,3), 8*num_inst, P_duality_gap_y);
  SS_SCRATCH_READ(getBankedOffset(0,3), 8*num_inst, P_duality_gap_E);
  
  SS_CONST(P_duality_gap_b, b, 1);
  SS_2D_CONST(P_duality_gap_const, 2, num_inst-1, 1, 1, 1);

  SS_DMA_WRITE(P_duality_gap_dgap, 8, 8, 1, &duality_gap);
  SS_WAIT_ALL();
  // cout << "Duality calc done\n";
}

void kernel_err_update(long tid, int i, int j, double diff1, double diff2, double y1, double y2)
{
  
  if(data_val[i].size()==0 || data_val[j].size()==0)  return;

  int val1_offset = getLinearAddr(getLinearOffset(0,4));
  int ind1_offset = getLinearAddr(getLinearOffset(1,4));
  int val2_offset = getLinearAddr(getLinearOffset(2,4));
  int ind2_offset = getLinearAddr(getLinearOffset(3,4));


  // int num_inst = M;
  int num_inst = M/NUM_THREADS;

  int start_id = tid*num_inst;
  int end_id = start_id+num_inst;

  double gauss_var = -1/(2*sigma*sigma); // double to fix
  // int m=1;
  SS_CONFIG(ksvm_config, ksvm_size);

  SS_CONST(P_ksvm_gauss_var, DOUBLE_TO_FIX(gauss_var), num_inst);
  SS_CONST(P_ksvm_alpha1, diff1, num_inst);
  SS_CONST(P_ksvm_alpha2, diff2, num_inst);
  SS_CONST(P_ksvm_y1, y1, num_inst);
  SS_CONST(P_ksvm_y2, y2, num_inst);

  // banked scratch read
  SS_SCRATCH_READ(getBankedOffset(0,3), 8*num_inst, P_ksvm_old_E);
  
  // it would be better if we get this from memory
  // SS_SCRATCH_READ(getLinearAddr(0), (data_ptr[end_id]-data_ptr[start_id])*4 ,P_ksvm_a_val);
  // SS_SCRATCH_READ(getLinearAddr(getLinearOffset(1,2)) ,(data_ptr[end_id]-data_ptr[start_id])*4 ,P_ksvm_a_ind);
  
  // broadcast_inst(tid, i, j);
  for(int k=0; k<num_inst; ++k) {
	if(data_val[k].size()==0)
	  continue;

    // TODO: could do data_val[k][0].size()

    SS_SCRATCH_READ(val1_offset, 8*(data_val[i].size()/2), P_ksvm_b_val);
    SS_SCRATCH_READ(ind1_offset, 8*(data_ind[i].size()/2), P_ksvm_b_ind);
    SS_SCRATCH_READ(val2_offset, 8*(data_val[j].size()/2), P_ksvm_c_val);
    SS_SCRATCH_READ(ind2_offset, 8*(data_ind[j].size()/2), P_ksvm_c_ind);
    // SS_DMA_READ(&data_val[i][0], 8, 8, data_val[i].size()/2, P_ksvm_b_val);
    // SS_DMA_READ(&data_ind[i][0], 8, 8, data_val[i].size()/2, P_ksvm_b_ind);
    // SS_DMA_READ(&data_val[j][0], 8, 8, data_val[j].size()/2, P_ksvm_c_val);
    // SS_DMA_READ(&data_ind[j][0], 8, 8, data_val[j].size()/2, P_ksvm_c_ind);
    
    SS_DMA_READ(&data_ind[k][0], 8, 8, data_val[k].size()/2, P_ksvm_a_ind);
    SS_DMA_READ(&data_val[k][0], 8, 8, data_val[k].size()/2, P_ksvm_a_val);
  }

  // write this in banked scratch
  SS_SCR_WRITE(P_ksvm_E, num_inst*8, getBankedOffset(0,3));
 
  SS_WAIT_ALL();
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

    // Select new i and j such that E[i] is max and E[j] is min do in CGRA
    for(int k=0; k<M; ++k){
      if(E[k]>E[i])
        i=k;
      if(E[k]<E[j])
        j=k;
    }
	// std::cout << "i: " << i << " j: " << j << std::endl;
    SS_CONFIG(eta_config, eta_size);
    load_ij_linear_scratch(i,j);

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
	// cout << L << " " << H << endl;
    // if(L==H) continue;
    double inter_prod = 0, norm1 = 0, norm2 = 0;
	// cout << "Sent for eta calculation\n";
    
    eta_calc(tid, i, j, inter_prod, norm1, norm2);
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

	// cout << "Sent for kernel err calculation\n";
    kernel_err_update(tid, i, j, diff1, diff, y[i], y[j]);

    duality_gap = 0;
	// cout << "Sent for duality gap calculation\n";
    calc_duality_gap(tid, b, duality_gap);

  }
}


void load_linear_scratch(long tid) {

  int val_offset = getLinearAddr(getLinearOffset(0,2));
  int ind_offset = getLinearAddr(getLinearOffset(1,2));

  int n_inst = M/NUM_THREADS;
  for(int i=tid*n_inst; i<(tid+1)*n_inst; ++i) {
    SS_DMA_SCRATCH_LOAD(&data_val[i][0], 4, 4, data_val[i].size(), val_offset);
    SS_DMA_SCRATCH_LOAD(&data_ind[i][0], 4, 4, data_ind[i].size(), val_offset);
    val_offset += data_val[i].size()*4;
    ind_offset += data_ind[i].size()*4;
  }
  SS_WAIT_SCR_WR();
  SS_WAIT_ALL();
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

   SS_CONFIG(eta_config, eta_size);
   load_linear_scratch(tid);
   // train(tid);
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

  FILE *m1_file;
  char lineToRead[5000];
  string str(file);

  m1_file = fopen(str.c_str(), "r");

  // m1_file = fopen("datasets/small_adult.data", "r");
  // m1_file = fopen("datasets/very_small.data", "r");
  printf("Start reading matrix1\n");

  data_ptr[0]=0;
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

    // padding, FIXME: what could dummy values be!
    if(data_ind[inst_id].size()%2!=0) {
      data_val[inst_id].pop_back();
      data_ind[inst_id].pop_back();
    }

    // push sentinal in the data (because we still use 64-bit sentinal here)
    for(int i=0; i<2; ++i) {
      data_val[inst_id].push_back(0);
      data_ind[inst_id].push_back(SENTINAL32);
    }

    inst_id++;
    data_ptr[inst_id] = data_ptr[inst_id-1] + data_val[inst_id-1].size();
  }

  printf("Finished reading input data\n");


  // SS_CONFIG(eta_config, eta_size);
  // load_linear_scratch(0);
  // train(0);
  begin_roi();
  train(0);
  end_roi();
  sb_stats();
   
/*
  assert(NUM_THREADS<cores);
  
  // Barrier initialization
  if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }

  pthread_t threads[NUM_THREADS];
  int rc;
  long t;
  for(t=0;t<NUM_THREADS;t++){
    printf("In main: creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);     
	if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
	  return 0;
    }
  }
  
  for(int i = 0; i < NUM_THREADS; ++i) {
    if(pthread_join(threads[i], NULL)) {
  	printf("Could not join thread %d\n", i);
      return -1;
    }
  }
  */

  // train();
  printf("svm training done\n");
 
  return 0;
}
