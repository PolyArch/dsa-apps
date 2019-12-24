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
#include "ss-intrin/ss_insts.h"
#include "../../../../common/include/sim_timing.h"
#include "../../../../common/include/net_util_func.h"
#include <inttypes.h>
#define NUM_THREADS	4
#define EIE_WIDTH 1

using namespace std;

// dense
uint16_t activations[M];
uint16_t counter[M];

// sparse (4096x1) Mx1
vector<uint16_t> act_val;
vector<uint16_t> act_ind;

// CSR format (NxM)
vector<uint16_t> wgt_val[N];
vector<uint16_t> wgt_ind[N];
uint16_t wgt_ptr[N];


uint16_t out_vec[N];

void count_required_multiplications() {
  int outer_count=0;
  // checking for only 1 row
  int n=0;
  for(int i=0; i<EIE_WIDTH; ++i) {
  int ind1=0, ind2=0;
  while(ind1<act_ind.size() && ind2<wgt_ind[i].size()) {
    if(act_ind[ind1]==wgt_ind[i][ind2]) {
      outer_count++;
      ++ind1; ++ind2;
    } else if(act_ind[ind1]<wgt_ind[i][ind2]) {
      ++ind1;
    } else {
      ++ind2;
    }
  }
  }
  // cout << "Required multiplications is: " << count << endl;
  cout << "Multiplications required is: " << outer_count << endl;
}


void load_linear_scratchpad(long tid) {
  int ncol = EIE_WIDTH;
  int start_col = ncol*tid;
  int end_col = start_col+ncol;
 
  // unsigned nweight_load = wgt_ptr[end_col] - wgt_ptr[start_col];
  // assert(nweight_load<4096); // 1 half can have only those many elements
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
// const int port_wv[4] = {P_eie_wval0,P_eie_wval1,P_eie_wval2,P_eie_wval3};
// const int port_wi[4] = {P_eie_wind0,P_eie_wind1,P_eie_wind2,P_eie_wind3};
const int port_wv[4] = {P_eie_wval0,1,2,3};
const int port_wi[4] = {P_eie_wind0,1,2,3};



// calculating inner product of number of columns = vector width (multiple
// columns could overlap variable sized columns)
void mv_merged(long tid) {
 
  int ncol = EIE_WIDTH;

  int i = tid*ncol;
  // writing 2-byte dot products
  SS_SCR_WRITE(P_eie_out_val, 2*ncol, i*2);

  // extra multiplications which are required at end
  // SS_DCONST(P_eie_aval, 0, 1, T16);
  // SS_DCONST(port_wv[0], 0, 1, T16);

  SS_CONST(P_eie_aval, 0, 1);
  SS_CONST(port_wv[0], 0, 1);

  // int stride=0; int scr_offset = getLinearOffset(1,2);

  // TODO: reading sparse activations from memory/banked scratchpad?
  SS_DMA_READ(&act_ind[0], 2, 2, act_ind.size(), P_eie_aind);
  // SS_DMA_READ(&act_ind[0], 2, 2, act_val.size(), P_eie_aind);
  SS_CONFIG_INDIRECT(T16, T16, 2);
  SS_INDIRECT(P_eie_match_ind1, &act_val[0], act_val.size(), P_eie_aval);

  // access corresponding columns from the weight matrix (actiation is reused
  // EIE_WIDTH multiple times)
  for(int r=0; r<EIE_WIDTH; ++r) {
    // SS_DMA_READ(&wgt_val[i+r][0], 2, 2, wgt_val[i+r].size(), port_wv[r]);
    SS_DMA_READ(&wgt_ind[i+r][0], 2, 2, wgt_ind[i+r].size(), port_wi[r]);
    SS_CONFIG_INDIRECT(T16, T16, 2);
    SS_INDIRECT(P_eie_match_ind2, &wgt_val[i+r][0], wgt_val[i+r].size(), port_wv[r]);
  }
 
  // sentinel required for the lists to complete
  // SS_DCONST(P_eie_aind, SENTINAL16-1, 1, T16);
  SS_CONST(P_eie_aind, SENTINAL16-1, 1);

  for(int r=0; r<EIE_WIDTH; ++r) {
    SS_CONST(port_wi[r], SENTINAL16-1, 1);
    // SS_DCONST(port_wi[r], SENTINAL16-1, 1, T16);
  }
  uint16_t x;
  SS_RECV(P_eie_done, x);
  SS_RESET();
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

  count_required_multiplications();

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
