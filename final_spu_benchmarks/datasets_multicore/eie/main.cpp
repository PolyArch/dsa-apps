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
// #include "small_sp.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include <inttypes.h>
#define NUM_THREADS	2
#define EIE_WIDTH 4

using namespace std;

// #define N 9216
// #define M 4096

vector<uint16_t> act_val;
vector<uint16_t> act_ind;

// dense
uint16_t activations[M];
uint16_t counter[M];

vector<uint16_t> wgt_val[N];
vector<uint16_t> wgt_ind[N];
uint16_t wgt_ptr[N];

uint16_t out_vec[N];

// Barrier variable
pthread_barrier_t barr;

// ISSUE1: for local read, address has to be local -- mapping would help here??
// Each core will execute 1 merge (all activations read into 1st core from DRAM
// and then -- it broadcasts the output value; weights on the network from linear spad)
void mv_merged(long tid) {
 
  // for only 1 broadcast, ncol should be equal to the vec_width
  int ncol = EIE_WIDTH;
  int start_col = ncol*tid;
  int end_col = start_col+ncol;
  
  unsigned nweight_load = wgt_ptr[end_col] - wgt_ptr[start_col];
  SB_CONFIG(eie_config,eie_size);
 
  SB_DMA_SCRATCH_LOAD(&wgt_ind[start_col][0], 2, 2, nweight_load, 0);
  SB_DMA_SCRATCH_LOAD(&wgt_val[start_col][0], 2, 2, nweight_load, getLinearAddr(getLinearOffset(1,2))); 
  SB_WAIT_SCR_WR();
 
  // write to dma, sure?
  SB_DMA_WRITE(P_eie_out_val, 2, 2, ncol, &out_vec[0]);
  int i = tid;
  int stride=0; int scr_offset = getLinearAddr(getLinearOffset(1,2));

  // col1
  SB_SCRATCH_READ(stride, wgt_val[i].size()*2, P_eie_wind0);
  SB_SCRATCH_READ(scr_offset+stride, wgt_ind[i].size()*2, P_eie_wval0);
  SB_CONST(P_eie_wind0, SENTINAL16, 1);
  SB_CONST(P_eie_wval0, 0, 1);

  // col2
  stride += wgt_val[i].size()*2;
  SB_SCRATCH_READ(stride, wgt_val[i+1].size()*2, P_eie_wind1);
  SB_SCRATCH_READ(scr_offset+stride, wgt_ind[i+1].size()*2, P_eie_wval1);
  SB_CONST(P_eie_wind1, SENTINAL16, 1);
  SB_CONST(P_eie_wval1, 0, 1);

  // col3
  stride += wgt_val[i+1].size();
  SB_SCRATCH_READ(stride, wgt_val[i+2].size()*2, P_eie_wind2);
  SB_SCRATCH_READ(scr_offset+stride, wgt_ind[i+2].size()*2, P_eie_wval2);
  SB_CONST(P_eie_wind2, SENTINAL16, 1);
  SB_CONST(P_eie_wval2, 0, 1);

  // col4
  stride += wgt_val[i+2].size();
  SB_SCRATCH_READ(stride, wgt_val[i+3].size()*2, P_eie_wind3);
  SB_SCRATCH_READ(scr_offset + stride, wgt_ind[i+3].size()*2, P_eie_wval3);
  SB_CONST(P_eie_wind3, SENTINAL16, 1);
  SB_CONST(P_eie_wval3, 0, 1);
  
  SB_WAIT_ALL(); 
}

void sparsify(){
  uint64_t mask = 0;
  addDest(mask, 1);
  int vec_width=8;
  int pad_size = M%vec_width;
  
  SB_CONFIG(sparsify_config, sparsify_size);

  SB_DMA_READ(&activations[0], 2, 2, M, P_sparsify_A);
  SB_DMA_READ(&counter[0], 2, 2, M, P_sparsify_B);
  
  SB_CONST(P_sparsify_sentinal, SENTINAL16, M+2); // max M times needed

  // has to be non-zero to be sent from here
  SB_CONST(P_sparsify_A, SENTINAL16, pad_size+vec_width); 
  SB_CONST(P_sparsify_B, SENTINAL16, pad_size+vec_width);

  // number of elements according to the port width
  SB_REM_PORT(P_sparsify_val, M, mask, P_eie_aval);
  SB_REM_PORT(P_sparsify_ind, M, mask, P_eie_aind);

  uint16_t temp;
  SB_RECV(P_sparsify_signal, temp);
  SB_RESET();
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
   if(tid==0) sparsify();
   else mv_merged(tid-1);
   end_roi();
   sb_stats();
   // pthread_exit(NULL);
   return NULL;
}

void check_correctness() {
  for(int i=0; i<N; ++i) {
	cout << "\tout_val: " << out_vec[i];
  }
}

int main(){

  // Reading dense activations
  char lineToRead[5000];

  string str(dense_act_file);
  FILE *dense_act_file2 = fopen(str.c_str(), "r");
  // FILE *dense_act_file2 = fopen("datasets/very_small/dense_act.data", "r");
  int ind=0;
  
  
  while(fgets(lineToRead, 5000, dense_act_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> activations[ind];
	// cout << " val: " << activations[ind] << "\n";
	ind++;
  } 
  fclose(dense_act_file2);
  printf("Done reading dense activations\n");


  // setting the counter
  for(int i=0; i<M; ++i) {
	counter[i]=i;
  }

  printf("Done setting counter\n");

  str = wgt_ptr_file;
  FILE *wgt_ptr_file2 = fopen(str.c_str(), "r");
  // FILE *wgt_ptr_file = fopen("datasets/pyfc6_wgt_ptr.txt", "r");
 
  ind=0;
  printf("Start reading wgt ptr\n");
  
  while(fgets(lineToRead, 5000, wgt_ptr_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> wgt_ptr[ind];
	ind++;
  }  
  fclose(wgt_ptr_file2);

  str = wgt_val_file;
  cout << wgt_val_file << "\n";
  FILE *wgt_val_file2 = fopen(str.c_str(), "r");
  // FILE *wgt_val_file = fopen("datasets/pyfc6_wgt_val.txt", "r");
 
  ind=0; int k=0;
  printf("Start reading wgt val\n");
  
  while(fgets(lineToRead, 5000, wgt_val_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	float x;

	iss >> x;
	// cout << "k: " << k << " x: " << x << endl;
	wgt_val[k].push_back(DOUBLE_TO_FIX(x));
	ind++;
	// FIXME: confirm this
	if(ind==wgt_ptr[k]) {
	  k++;
	}
  }  
  fclose(wgt_val_file2);

  str = wgt_ind_file;
  FILE *wgt_ind_file2 = fopen(str.c_str(), "r");
  // FILE *wgt_ind_file = fopen("datasets/pyfc6_wgt_ind.txt", "r");
 
  ind=0; k=0;
  printf("Start reading wgt_ind activations\n");
  
  while(fgets(lineToRead, 5000, wgt_ind_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;

	iss >> x;
	wgt_ind[k].push_back(x);
	ind++;
	// FIXME: confirm this
	if(ind==wgt_ptr[k]) {
	  k++;
	}
  }  
  fclose(wgt_ind_file2);

  // sparsify();

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
  
  if(print_result) {
    check_correctness();
  }
  return 0;
}
