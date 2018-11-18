#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <assert.h>
#include <vector>
#include <pthread.h>
#include "eie.dfg.h"
#include "sparsify.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#define NUM_THREADS	2

// #define sentinal (SENTINAL16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48)
#define sentinal SENTINAL

using namespace std;

#define N 9216
#define M 4096

vector<uint16_t> act_val;
vector<uint16_t> act_ind;

// dense
uint16_t activations[M];
uint16_t *counter;

// vector<float> fwgt_val[N];
vector<uint16_t> wgt_val[N];
vector<uint16_t> wgt_ind[N];
uint16_t wgt_ptr[N];

// uint64_t *out_vec;
uint16_t out_vec[N];

// Barrier variable
pthread_barrier_t barr;

// Each core will execute 1 merge (all activations read into 1st core from DRAM and then
// it broadcasts the output value; weights on the network from linear spad)
void mv_merged(long tid) {
  // int ptr1=0, end1=0;
  unsigned row_size=0;
  // std::cout << "reached till here\n";
  // load into scratchpads in main
 
  int ncol = N/(4*NUM_THREADS); // 4 because of the vec width
  int start_col = ncol*tid;
  int end_col = start_col+ncol;
  // each thread id would have N/63 columns 
  
  unsigned nweight_load = wgt_ptr[end_col] - wgt_ptr[start_col];
  // cout << "Number of weights are: " << nweight_load << "\n";
 
  SB_CONFIG(eie_config,eie_size);

  // since it is now using a port; either we should add a condition not to reset those ports
  // ISSUE1: for local read, address has to be local -- mapping would help here??
  // SB_DMA_SCRATCH_LOAD(&wgt_merged_col_ind[wgt_merged_row_ptr[start_col]], 8, 8, nweight_load, (tid << 16) | 0);
  // SB_DMA_SCRATCH_LOAD(&wgt_merged_val[wgt_merged_row_ptr[start_col]], 8, 8, nweight_load, (tid << 16) | (4095)); // fix this offset
  SB_DMA_SCRATCH_LOAD(&wgt_ind[start_col][0], 2, 2, nweight_load, 0);
  SB_DMA_SCRATCH_LOAD(&wgt_val[start_col][0], 2, 2, nweight_load, nweight_load); // TODO: set correct offset
 
  SB_WAIT_SCR_WR();
 
  SB_DMA_WRITE(P_eie_out_val, 8, 8, ncol, &out_vec[0]); // write to the DMA? or are we doing multiple layers
  // cout << "tid: " << tid << " ncol: " << ncol << "\n";
  for (int i=tid; i<tid+ncol; ++i){ // work on a subset of columns
	// if(end1-ptr1<=0)
	//   continue;
	// row_size = (end1-ptr1);
	row_size = wgt_val[i].size();
    
	// SB_SCRATCH_READ((tid << 16) | 0, row_size*8, P_eie_wind);
	SB_SCRATCH_READ(0, row_size*2, P_eie_wind);
	SB_CONST(P_eie_wind, sentinal, 1);
	// SB_SCRATCH_READ((tid << 16) | 4095, row_size*8, P_eie_wval);
	SB_SCRATCH_READ(nweight_load, row_size*2, P_eie_wval);
	SB_CONST(P_eie_wval, 0, 1);

	// double read instead of broadcast?
    SB_DMA_READ(&act_ind[0], 2, 2, act_ind.size(), P_eie_aind); // being broadcasted from core 0
    SB_DMA_READ(&act_val[0], 2, 2, act_val.size(), P_eie_aval); // being broadcasted from core 0
  }
  // error in this wait all
  SB_WAIT_ALL(); 
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

   begin_roi();
   mv_merged(tid);
   // mv_single(tid);
   end_roi();
   sb_stats();
   // pthread_exit(NULL);
   return NULL;
}

// FIXME: padding
void sparsify(){
  uint16_t a[M];
  uint16_t b[M];
  int values = M/4;
  uint64_t mask = (((uint64_t)1)<<63)-1;
  
  SB_CONFIG(sparsify_config, sparsify_size);

  SB_DMA_READ(&activations[0], 8, 8, values, P_sparsify_A);
  SB_DMA_READ(&counter[0], 8, 8, values, P_sparsify_B);
  SB_CONST(P_sparsify_size, N*M-1, values*4); // should be a 16-bit input

  // sentinals (not allowing to push this constant)
  SB_CONST(P_sparsify_A, sentinal, 1);
  SB_CONST(P_sparsify_B, sentinal, 1);

  SB_STRIDE(8,8);
  SB_DMA_WRITE_SIMP(P_sparsify_val, values, &a[0]);
  SB_DMA_WRITE_SIMP(P_sparsify_ind, values, &b[0]);

  // SB_DMA_WRITE_SHF16(P_sparsify_val, 8, 8, values, &a[0]);
  // SB_DMA_WRITE_SHF16(P_sparsify_ind, 8, 8, values, &b[0]);

  // how would I copy only 16-bits? (only possible now by accumulate)
  // SB_REM_PORT(P_sparsify_val, values, mask, P_eie_aval);
  // SB_REM_PORT(P_sparsify_ind, values, mask, P_eie_aind);

  uint16_t temp;
  SB_RECV(P_sparsify_signal, temp);
  SB_RESET();
  SB_WAIT_ALL();
}

void check_correctness() {
  for(int i=0; i<N; ++i) {
	cout << "\tout_val: " << out_vec[i];
  }
}

uint64_t merge_bits(uint16_t a, uint16_t b, uint16_t c, uint16_t d){
  uint64_t ret = (a | (b  & 0xFFFFFFFFFFFFFFFF) << 16 | (c & 0xFFFFFFFFFFFFFFFF) << 32 | (d & 0xFFFFFFFFFFFFFFFF) << 48);
  return ret;
}


int main(){

  // Reading dense activations
  char lineToRead[5000];

  FILE *dense_act_file = fopen("datasets/pyfc6_dense_act_file.txt", "r");
  int ind=0;
  
  printf("Start reading dense activations\n");
  
  while(fgets(lineToRead, 5000, dense_act_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> activations[ind];
	cout << "ind: " << counter[ind] << " val: " << activations[ind] << "\n";
	ind++;
  } 
  fclose(dense_act_file);
  printf("Done reading dense activations\n");

  FILE *wgt_ptr_file = fopen("datasets/pyfc6_wgt_ptr.txt", "r");
 
  ind=0;
  printf("Start reading dense activations\n");
  
  while(fgets(lineToRead, 5000, wgt_ptr_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> wgt_ptr[ind];
	ind++;
  }  
  fclose(wgt_ptr_file);


  FILE *wgt_val_file = fopen("datasets/pyfc6_wgt_val.txt", "r");
 
  ind=0; int k=0;
  printf("Start reading dense activations\n");
  
  while(fgets(lineToRead, 5000, wgt_val_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	float x;

	iss >> x;
	wgt_val[k].push_back(DOUBLE_TO_FIX(x));
	// need pointer?
	cout << "ind: " << counter[ind] << " val: " << activations[ind] << "\n";
	ind++;
	// FIXME: confirm this
	if(ind==wgt_ptr[k]) {
	  k++;
	}
  }  
  fclose(wgt_val_file);

  FILE *wgt_ind_file = fopen("datasets/pyfc6_wgt_ind.txt", "r");
 
  ind=0; k=0;
  printf("Start reading dense activations\n");
  
  while(fgets(lineToRead, 5000, wgt_ind_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;

	iss >> x;
	wgt_ind[k].push_back(x);
	// need pointer?
	cout << "ind: " << counter[ind] << " val: " << activations[ind] << "\n";
	ind++;
	// FIXME: confirm this
	if(ind==wgt_ptr[k]) {
	  k++;
	}
  }  
  fclose(wgt_ptr_file);

  sparsify();

  /*
  assert(NUM_THREADS<C);
  
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
  
  if(print_result) {
    check_correctness();
  }

  return 0;
}
