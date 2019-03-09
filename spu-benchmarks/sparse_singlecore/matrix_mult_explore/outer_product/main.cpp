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
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
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

// CSC format
// vector<uint16_t> wgt_val[M];
// vector<uint16_t> wgt_ind[M];
vector<uint16_t> wgt_val;
vector<uint16_t> wgt_ind;
uint16_t wgt_ptr[M+1];


uint16_t out_vec[N];

// Assuming 16-bit compute only now (4-bit could improve things)
void mv() {

  // cout << "Act size: " << act_val.size() << endl;
  // cout << "Wgt size: " << wgt_val.size() << endl;
  /*int asize = act_val.size();
  for(int i=0; i<asize; ++i) {
    cout << "Activation indices: " << act_ind[i] << endl;
    cout << "Correpsonding ptrs: " << wgt_ptr[act_ind[i]] << " " << wgt_ptr[act_ind[i]+1] << endl; 
  }*/

  // read activations -- move to linear scratchpad later
  SS_VREPEAT_PORT(P_eie_row_size2);
  SS_DMA_READ(&act_val[0], 2, 2, act_val.size(), P_eie_aval);
 
  SS_DMA_READ(&act_ind[0], 2, 2, act_ind.size(), P_IND_1);
  SS_CONFIG_INDIRECT1(T16,T16,2,1);
  SS_INDIRECT(P_IND_1, &wgt_ptr[0], act_ind.size(), P_eie_offset_list);

  // use 2D_SCR when weights are in scratchpad
  SS_CONFIG_INDIRECT1(T16,T16,2,1);
  SS_INDIRECT_2D(P_eie_start_ind, &wgt_val[0], act_val.size(), 4, 2, P_eie_row_size1, P_eie_wval); // in reality, this is also unknown (should be M)

  // since I know, let's get it working using the actual size (actually wgt we will know)
  // FIXME:CHECKME sum up the value in the output
  SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  SS_ATOMIC_SCR_OP(P_eie_addr, P_eie_val, 0, wgt_val.size()/2, 0);
  
  // need an and signal to reset: wait on everything except atomic scr
  // wait until ports are free (don't think of streams)
  // SS_WAIT_COMPUTE(); // Let's see what happens here
  // SS_RESET();
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

  FILE *wgt_file = fopen("input_csc_weights.data", "r");

  printf("Start reading wgt ptr\n");
  wgt_ptr[0]=0;
  int prev_index=0, count=0;
  
  while(fgets(lineToRead, 5000, wgt_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    int first_index, second_index, val;

	iss >> first_index >> second_index >> val;
    // wgt_ind.push_back(first_index);
    // cout << "Pushing value: " << val << endl;
    wgt_val.push_back(val);
    wgt_val.push_back(first_index);
    // wgt_ind[second_index].push_back(first_index);
    // wgt_val[second_index].push_back(val);

    // FIXME: confirm this is correct
    if(second_index!=prev_index) {
      wgt_ptr[second_index] = count;
      for(int i=second_index-1; i>prev_index; --i) {
        wgt_ptr[i] = count;
      }
      prev_index=second_index;
    }
    count++;
  }  
  wgt_ptr[M]=count;
  fclose(wgt_file);

  /*for(int i=0; i<4; ++i) {
    cout << "Address of wgt_val at i: " << i << " is: " << &wgt_val[i] << " and value: " << wgt_val[i] << endl;
  }*/

  printf("Finished reading wgt file\n");

  for(int i=0; i<N; ++i) {
    out_vec[i]=0;
  }

   // load_linear_scratchpad(0);
   // TODO: works only when it fits in there
   // load_banked_scratchpad();
   SS_CONFIG(eie_config,eie_size);
   // mv_merged(0);
   // mv();
   begin_roi();
   mv();
   end_roi();
   sb_stats();
 
  return 0;
}
