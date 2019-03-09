#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <assert.h>
#include <vector>
#include <pthread.h>
#include <cstring>
#include <map>
#include "eie.dfg.h"
#include "/home/vidushi/ss-stack/riscv-opcodes/ss_insts.h"
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

// CSR format
vector<uint16_t> wgt_val[N];
vector<uint16_t> wgt_ind[N];
uint16_t wgt_ptr[N];


uint16_t out_vec[N];


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


// map from the row_id to the port id (store those variable values here)
const int port_wv[4] = {P_eie_wval0,P_eie_wval1,P_eie_wval2,P_eie_wval3};
const int port_wi[4] = {P_eie_wind0,P_eie_wind1,P_eie_wind2,P_eie_wind3};


void mv_merged(long tid) {
 
  int ncol = EIE_WIDTH;

  int i = tid*ncol;
  SS_SCR_WRITE(P_eie_out_val, 2*ncol, i*2);
  // int stride=0; int scr_offset = getLinearOffset(1,2);

  // TODO: reading sparse activations from memory/banked scratchpad?
  SS_DMA_READ(&act_val[0], 2, 2, act_val.size(), P_eie_aval);
  SS_DMA_READ(&act_ind[0], 2, 2, act_val.size(), P_eie_aind);

  for(int r=0; r<4; ++r) {
    SS_DMA_READ(&wgt_val[i+r][0], 2, 2, wgt_val[i+r].size(), port_wv[r]);
    SS_DMA_READ(&wgt_ind[i+r][0], 2, 2, wgt_ind[i+r].size(), port_wi[r]);
  }
 
  SS_CONST(P_eie_aind, SENTINAL16, 1);
  SS_CONST(P_eie_aval, 0, 1);

  for(int r=0; r<4; ++r) {
    SS_CONST(port_wi[r], SENTINAL16, 1);
    SS_CONST(port_wv[r], 0, 1);
  }
  SS_WAIT_ALL(); 
}

void read_act() {


  char lineToRead[5000];
  
  FILE *act_file = fopen("input_activations.data", "r");
 
  printf("Start reading activations\n");
  while(fgets(lineToRead, 5000, act_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int ind, val;

	iss >> ind >> val;
	act_ind.push_back(ind);
    act_val.push_back(val);
  }  
  fclose(act_file);
 
  printf("Done reading activations\n");
}

int main(){

  // Reading dense activations
  char lineToRead[5000];

  read_act();
  printf("Done reading sparse activations\n");

  FILE *wgt_file = fopen("input_weights.data", "r");

  printf("Start reading wgt ptr\n");
  
  while(fgets(lineToRead, 5000, wgt_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    int first_index, second_index, val;

	iss >> first_index >> second_index >> val;
    wgt_ind[first_index].push_back(second_index);
    wgt_val[first_index].push_back(val);
  }  
  fclose(wgt_file);

  printf("Finished reading wgt file\n");

  for(int i=0; i<N; ++i) {
    out_vec[i]=0;
  }

   // load_linear_scratchpad(0);
   // TODO: works only when it fits in there
   // load_banked_scratchpad();
   SS_CONFIG(eie_config,eie_size);
   mv_merged(0);
   begin_roi();
   mv_merged(0);
   end_roi();
   sb_stats();
 
  return 0;
}
