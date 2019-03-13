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
uint16_t wgt_ptr[M+2];


uint16_t out_vec[N];

// 12132 = 16384
// Assuming 16-bit compute only now (4-bit could improve things)
void mv() {

  // read activations -- move to linear scratchpad later
  SS_VREPEAT_PORT(P_eie_row_size2);
  SS_DMA_READ(&act_val[0], 2, 2, act_val.size(), P_eie_aval);
 
  SS_DMA_READ(&act_ind[0], 2, 2, act_ind.size(), P_IND_1);
  SS_CONFIG_INDIRECT1(T16,T16,2,1);
  SS_INDIRECT(P_IND_1, &wgt_ptr[0], act_ind.size(), P_eie_offset_list);

  // use 2D_SCR when weights are in scratchpad
  // FIXME:this value should not be 0
  // SS_CONFIG_INDIRECT1(T16,T16,2,1);
  SS_CONFIG_INDIRECT1(T16,T16,4,1); // multiplier for offset
  SS_INDIRECT_2D(P_eie_start_ind, &wgt_val[0], act_val.size(), 4, 2, P_eie_row_size1, P_eie_wval); // in reality, this is also unknown (should be M)

  // FIXME: this should be sentinal later
  // SS_CONST(P_eie_wval, 0, 8); // this doesn't make any sense
  // SS_CONST(P_eie_aval, 0, 1);

  // FIXME:CHECKME sum up the value in the output
  SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  SS_ATOMIC_SCR_OP(P_eie_addr, P_eie_val, 0, wgt_val.size()/2, 0);  
 
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
    // cout << "Ind: " << ind << " val: " << val << endl;
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
      // to make the column sizes multiple of 4
      int prev_col_size = count-wgt_ptr[prev_index];
      int pad_size = (int)prev_col_size%4; // vec width
      if(pad_size!=0) {
        pad_size = 4 - pad_size;
        for(int i=0; i<pad_size; ++i) {
          wgt_val.push_back(1);
          wgt_val.push_back(1);
          // FIXME: later, this should be 0 and later sentinal
          // wgt_val.push_back(0);
          // wgt_val.push_back(0);
 
        }
        count += pad_size;
      }
      wgt_ptr[second_index] = count;
      for(int i=second_index-1; i>prev_index; --i) {
        wgt_ptr[i] = count;
      }
      prev_index=second_index;
    }
    count++;
  }  

  int prev_col_size = count-wgt_ptr[prev_index];
  int pad_size = (int)prev_col_size%4; // vec width
  if(pad_size!=0) {
     pad_size = 4 - pad_size;
     for(int i=0; i<pad_size; ++i) {
       // wgt_val.push_back(0);
       // wgt_val.push_back(0);
       wgt_val.push_back(1);
       wgt_val.push_back(1);
     }
     count += pad_size;
   }
  wgt_ptr[M]=count;
  for(int i=M-1; i>prev_index; --i) {
    wgt_ptr[i] = count;
  }
    
  fclose(wgt_file);

  // preprocess_weights();

  /*cout << "Activation values: ";
  for(int i=0; i<4; ++i) {
    cout << act_val[i] << endl;
  }*/
  /*for(int i=0; i<M; ++i) {
    // cout << "Address of wgt_val at i: " << i << " is: " << &wgt_val[i] << " and value: " << wgt_val[i] << endl;
    cout << "Index pointer at column i: " << i << " is: " << wgt_ptr[i] << endl;
  }*/


  // Pre-processing here: 0 at the end of both col and row vector
  act_val.push_back(0);
  act_ind.push_back(M);
  for(int i=0; i<4; ++i) {
    wgt_val.push_back(0); // val
    wgt_val.push_back(M); // col-index
  }
  wgt_ptr[M+1] = wgt_ptr[M]+4;

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

/*void preprocess_weights() {
  int col_size=-1;
  int cur_pad_size=-1;
  int acc_pad_size=0;
  for(int i=0; i<M; ++i) { // for M columns in wgt
    col_size = wgt_ptr[i+1]-wgt_ptr[i];
    cur_pad_size = int(col_size%4);
    if(cur_pad_size!=0) cur_pad_size = 4-cur_pad_size;
    for(int j=0; j<cur_pad_size; ++j) { // both index and val are 0
      wgt_val.push_back(0);
      wgt_val.push_back(0);
    }
    acc_pad_size += cur_pad_size;
    wgt_ptr[i+1] += acc_pad_size;
  }
}*/
