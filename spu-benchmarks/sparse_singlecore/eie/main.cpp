#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <assert.h>
#include <vector>
#include <pthread.h>
#include <cstring>
#include "eie.dfg.h"
#include "sparsify.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include <inttypes.h>
#define NUM_THREADS	4
#define EIE_WIDTH 4

using namespace std;

// dense
uint16_t activations[M];
uint16_t counter[M];

// sparse
vector<uint16_t> act_val;
vector<uint16_t> act_ind;

vector<uint16_t> wgt_val[N];
vector<uint16_t> wgt_ind[N];
uint16_t wgt_ptr[N];

uint16_t out_vec[N];

// Barrier variable
pthread_barrier_t barr;

void load_linear_scratchpad(long tid) {
  int ncol = EIE_WIDTH;
  int start_col = ncol*tid;
  int end_col = start_col+ncol;
 
  unsigned nweight_load = wgt_ptr[end_col] - wgt_ptr[start_col];
  assert(nweight_load<4096); // 1 half can have only those many elements
  unsigned stride=0;

  for(int i=start_col; i<end_col; ++i) {
    SS_DMA_SCRATCH_LOAD(&wgt_ind[i][0], 2, 2, wgt_ind[i].size(), getLinearAddr(0+stride));
    SS_DMA_SCRATCH_LOAD(&wgt_val[i][0], 2, 2, wgt_val[i].size(), getLinearAddr(getLinearOffset(1,2)+stride)); 
    stride += wgt_ind[i].size();
  }

  // SS_DMA_SCRATCH_LOAD(&wgt_ind[start_col][0], 2, 2, nweight_load, getLinearAddr(0));
  // SS_DMA_SCRATCH_LOAD(&wgt_val[start_col][0], 2, 2, nweight_load, getLinearAddr(getLinearOffset(1,2))); 
  SS_WAIT_SCR_WR();
  SS_WAIT_ALL();
}


void load_banked_scratchpad() {

  SS_DMA_SCRATCH_LOAD(&act_ind[0], 2, 2, act_ind.size(), getBankedOffset(0,2));
  SS_DMA_SCRATCH_LOAD(&act_val[0], 2, 2, act_val.size(), getBankedOffset(1,2));
  SS_WAIT_ALL();

}


// ISSUE1: for local read, address has to be local -- mapping would help here??
// Each core will execute 1 merge (all activations read into 1st core from DRAM
// and then -- it broadcasts the output value; weights on the network from linear spad)
void mv_merged(long tid) {
 
  // for only 1 broadcast, ncol should be equal to the vec_width
  int ncol = EIE_WIDTH;
  // int start_col = ncol*tid;
  // int end_col = start_col+ncol;

  // write to dma, sure?
  int i = tid*ncol;
  // SS_DMA_WRITE(P_eie_out_val, 2, 2, ncol, &out_vec[i]);
  SS_SCR_WRITE(P_eie_out_val, 2*ncol, i*2);
  int stride=0; int scr_offset = getLinearOffset(1,2);

  // TODO: reading sparse activations from memory/banked scratchpad?
  SS_DMA_READ(&act_val[0], 2, 2, act_val.size(), P_eie_aval);
  SS_DMA_READ(&act_ind[0], 2, 2, act_val.size(), P_eie_aind);
  // SS_SCRATCH_READ(getBankedOffset(0,2), 2*act_val.size(), P_eie_aval);
  // SS_SCRATCH_READ(getBankedOffset(1,2), 2*act_ind.size(), P_eie_aind);
  // col1
  SS_DMA_READ(&wgt_val[i][0], 2, 2, wgt_val[i].size(), P_eie_wval0);
  SS_DMA_READ(&wgt_ind[i][0], 2, 2, wgt_ind[i].size(), P_eie_wind0);
  // SS_SCRATCH_READ(getLinearAddr(stride), wgt_val[i].size()*2, P_eie_wind0);
  // SS_SCRATCH_READ(getLinearAddr(scr_offset+stride), wgt_ind[i].size()*2, P_eie_wval0);
  // col2
  SS_DMA_READ(&wgt_val[i+1][0], 2, 2, wgt_val[i+1].size(), P_eie_wval1);
  SS_DMA_READ(&wgt_ind[i+1][0], 2, 2, wgt_ind[i+1].size(), P_eie_wind1);
 
  stride += wgt_val[i].size()*2;
  // SS_SCRATCH_READ(getLinearAddr(stride), wgt_val[i+1].size()*2, P_eie_wind1);
  // SS_SCRATCH_READ(getLinearAddr(scr_offset+stride), wgt_ind[i+1].size()*2, P_eie_wval1);
  // col3
  stride += wgt_val[i+1].size()*2;
  SS_DMA_READ(&wgt_val[i+2][0], 2, 2, wgt_val[i+2].size(), P_eie_wval2);
  SS_DMA_READ(&wgt_ind[i+2][0], 2, 2, wgt_ind[i+2].size(), P_eie_wind2);
 
  // SS_SCRATCH_READ(getLinearAddr(stride), wgt_val[i+2].size()*2, P_eie_wind2);
  // SS_SCRATCH_READ(getLinearAddr(scr_offset+stride), wgt_ind[i+2].size()*2, P_eie_wval2);
  // col4
  stride += wgt_val[i+2].size()*2;
  SS_DMA_READ(&wgt_val[i+3][0], 2, 2, wgt_val[i+3].size(), P_eie_wval3);
  SS_DMA_READ(&wgt_ind[i+3][0], 2, 2, wgt_ind[i+3].size(), P_eie_wind3);
 
  // SS_SCRATCH_READ(getLinearAddr(stride), wgt_val[i+3].size()*2, P_eie_wind3);
  // SS_SCRATCH_READ(getLinearAddr(scr_offset + stride), wgt_ind[i+3].size()*2, P_eie_wval3);
  SS_CONST(P_eie_wind0, SENTINAL16, 1);
  SS_CONST(P_eie_wval0, 0, 1);

  SS_CONST(P_eie_aind, SENTINAL16, 1);
  SS_CONST(P_eie_aval, 0, 1);

  SS_CONST(P_eie_wind1, SENTINAL16, 1);
  SS_CONST(P_eie_wval1, 0, 1);

  SS_CONST(P_eie_wind2, SENTINAL16, 1);
  SS_CONST(P_eie_wval2, 0, 1);

  SS_CONST(P_eie_wind3, SENTINAL16, 1);
  SS_CONST(P_eie_wval3, 0, 1);
 
  SS_WAIT_ALL(); 
}

void sparsify(){
  uint64_t mask = 0;
  for(int i=1; i<NUM_THREADS; ++i) {
    addDest(mask, i);
  }
  int vec_width=8;
  int pad_size = M%vec_width;
  
  SS_DMA_READ(&activations[0], 2, 2, M, P_sparsify_A);
  SS_DMA_READ(&counter[0], 2, 2, M, P_sparsify_B);
  
  // SS_CONST(P_sparsify_sentinal, SENTINAL16, M+2); // max M times needed
  // to debug
  SS_CONST(P_sparsify_sentinal, SENTINAL16, M+8); // max M times needed

  // has to be non-zero to be sent from here
  SS_CONST(P_sparsify_A, SENTINAL16, pad_size+vec_width); 
  SS_CONST(P_sparsify_B, SENTINAL16, pad_size+vec_width);

  // number of elements according to the port width
  SS_REM_PORT(P_sparsify_val, M+8, mask, P_eie_aval);
  SS_REM_PORT(P_sparsify_ind, M+8, mask, P_eie_aind);

  uint16_t temp;
  SS_RECV(P_sparsify_signal, temp);
  SS_RESET();
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

   if(tid!=0) {
     SS_CONFIG(eie_config,eie_size);
     load_linear_scratchpad(tid-1);
   } else {
     SS_CONFIG(sparsify_config, sparsify_size);
   }
   begin_roi();
   if(tid==0) {
     sparsify();
   }
   else {
     mv_merged(tid-1);
   }
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


void read_act() {


  char lineToRead[5000];
  string str(layer_name);
  
  char r[100] = "datasets/";
  char d[100] = "/act_index.txt";
  FILE *act_ind_file2 = fopen(strcat(strcat(r,str.c_str()),d), "r");
 
  printf("Start reading wgt_ind activations\n");
  while(fgets(lineToRead, 5000, act_ind_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;

	iss >> x;
	act_ind.push_back(x);
    act_val.push_back(3);
  }  
  fclose(act_ind_file2);
 
  printf("Done reading wgt_ind activations\n");
  /*
  char q[100] = "datasets/";
  char c[100] = "/act_val.data";
  FILE *wgt_val_file2 = fopen(strcat(strcat(q,str.c_str()),c), "r");
 
  ind=0; int k=0;
  printf("Start reading act val\n");
  
  
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
*/  

}

int main(){

  // Reading dense activations
  char lineToRead[5000];

  // string str(dense_act_file);
  string str(layer_name);
  /*
  char s[100] = "datasets/";
  char a[100] = "/dense_act.data";
  // FILE *dense_act_file2 = fopen(str.c_str(), "r");
  FILE *dense_act_file2 = fopen(strcat(strcat(s,str.c_str()),a), "r");
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
    // activations[i]=10*i;
	counter[i]=i;
  }

  printf("Done setting counter\n");
  */
  read_act();
  printf("Done reading sparse activations\n");

  // TODO: read sparse activations here

  // str = wgt_ptr_file;
  // FILE *wgt_ptr_file2 = fopen(str.c_str(), "r");
  // FILE *wgt_ptr_file = fopen("datasets/pyfc6_wgt_ptr.data", "r");
  char p[100] = "datasets/";
  char b[100] = "/wgt_ptr.data";
  FILE *wgt_ptr_file2 = fopen(strcat(strcat(p,str.c_str()),b), "r");
  // FILE *wgt_ptr_file2 = fopen(strcat(strcat(s,str.c_str()),b), "r");
 
  int ind=0;
  printf("Start reading wgt ptr\n");
  
  while(fgets(lineToRead, 5000, wgt_ptr_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> wgt_ptr[ind];
	ind++;
  }  
  fclose(wgt_ptr_file2);

  printf("Finished reading wgt ptr\n");
  // str = wgt_val_file;
  // cout << wgt_val_file << "\n";
  char q[100] = "datasets/";
  char c[100] = "/wgt_val.data";
  FILE *wgt_val_file2 = fopen(strcat(strcat(q,str.c_str()),c), "r");
 
  // FILE *wgt_val_file2 = fopen(str.c_str(), "r");
  // FILE *wgt_val_file = fopen("datasets/pyfc6_wgt_val.txt", "r");
 
  ind=0; int k=0;
  printf("Start reading wgt val\n");
 /* 
  
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
 */

  // str = wgt_ind_file;

  /*
  int n=0; // size
  for(int i=0; i<N; ++i) {
    if(i>0) {
      n = wgt_ptr[i]-wgt_ptr[i-1];
      wgt_val[i].resize(n);
      // wgt_ind[i].resize(n);
    } else {
      n = wgt_ptr[i];
      wgt_val[i].resize(n);
      // wgt_ind[i].resize(n);
    }
   
    for(int j=0; j<n; ++j) {
      wgt_ind[i][j]=j;
    }
    
  }
  */
  
  printf("Start reading wgt_ind activations\n");
  char r[100] = "datasets/";
  char d[100] = "/wgt_index.data";
  FILE *wgt_ind_file2 = fopen(strcat(strcat(r,str.c_str()),d), "r");
 

  // FILE *wgt_ind_file2 = fopen(str.c_str(), "r");
  // FILE *wgt_ind_file = fopen("datasets/pyfc6_wgt_ind.txt", "r");
 
  ind=0; k=0;
  
  while(fgets(lineToRead, 5000, wgt_ind_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;

	iss >> x;
	wgt_ind[k].push_back(x);
    wgt_val[k].push_back(3);
	ind++;
	// FIXME: confirm this
	if(ind==wgt_ptr[k]) {
	  k++;
	}
  }  
  fclose(wgt_ind_file2);
  

  for(int i=0; i<N; ++i) {
    out_vec[i]=0;
  }

   SS_CONFIG(eie_config,eie_size);
   // load_linear_scratchpad(0);
   // TODO: works only when it fits in there
   // load_banked_scratchpad();
   mv_merged(0);
   begin_roi();
   mv_merged(0);
   end_roi();
   sb_stats();
 
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
  
  if(print_result) {
    check_correctness();
  }
  */
  return 0;
}
