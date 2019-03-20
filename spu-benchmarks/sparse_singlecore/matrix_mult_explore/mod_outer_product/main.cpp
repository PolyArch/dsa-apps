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
int col_pad_size[M];


uint16_t out_vec[N];

// Assuming 16-bit compute only now (4-bit could improve things)
void mv() {

  // SS_DMA_READ(&act_val[0], 2, 2, act_val.size(), P_IND_3);
  SS_DMA_READ(&act_val[0], 2, 2, act_val.size(), P_eie_pass1);

  // This should be all done actually, right?
  SS_VREPEAT_PORT(P_eie_row_size2);
  SS_RECURRENCE(P_eie_pass2, P_eie_aval, act_val.size());
  // SS_RECURRENCE(P_IND_3, P_eie_aval, act_val.size());
 
  SS_DMA_READ(&act_ind[0], 2, 2, act_ind.size(), P_IND_1);
  SS_CONFIG_INDIRECT1(T16,T16,2,1);
  SS_INDIRECT(P_IND_1, &wgt_ptr[0], act_ind.size(), P_eie_offset_list);

  SS_CONFIG_INDIRECT1(T16,T16,4,1); // multiplier for offset
  SS_INDIRECT_2D(P_eie_start_ind, &wgt_val[0], act_val.size(), 4, 2, P_eie_row_size1, P_eie_wval);

  SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  SS_ATOMIC_SCR_OP(P_eie_addr, P_eie_val, 0, wgt_val.size()/2, 0);  
 
  uint16_t x;
  SS_RECV(P_eie_done, x);
  SS_RESET();
  SS_WAIT_ALL(); 
}

void count_required_multiplications() {
  int count=0;
  int outer_count=0;
  for(unsigned i=0; i<act_val.size(); ++i) {
    count += (wgt_ptr[i+1]-wgt_ptr[i]-col_pad_size[i]);
    outer_count += (wgt_ptr[i+1]-wgt_ptr[i]);
  }
  cout << "Required multiplications is: " << count << endl;
  cout << "Multiplications done in outer product is: " << outer_count << endl;
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

void read_weights() {
  char lineToRead[5000];

  FILE *wgt_file = fopen("input_csc_weights.data", "r");

  printf("Start reading wgt ptr\n");
  wgt_ptr[0]=0;
  int prev_index=0, count=0;
  
  while(fgets(lineToRead, 5000, wgt_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    int first_index, second_index, val;

	iss >> first_index >> second_index >> val;
    wgt_val.push_back(val);
    wgt_val.push_back(first_index);

    if(second_index!=prev_index) {
      wgt_ptr[second_index] = count;
      for(int i=second_index-1; i>prev_index; --i) {
        wgt_ptr[i] = count;
      }
      prev_index=second_index;
    }
    count++;
  }  

  wgt_ptr[M] = wgt_val.size()/2;

  for(int i=M-1; i>prev_index; --i) {
    wgt_ptr[i] = wgt_ptr[M];
  }
    
  fclose(wgt_file);
}

void print_data() {
  cout << "Activation values: ";
  for(unsigned i=0; i<act_val.size(); ++i) {
    cout << "Act val at i: " << i << " is: " << act_val[i] << endl;
  }
  for(unsigned i=0; i<wgt_val.size(); ++i) {
    cout << "Wgt val at i: " << i << " is: " << wgt_val[i] << endl;
  }
  for(int i=0; i<M+2; ++i) {
    cout << "Index pointer at column i: " << i << " is: " << wgt_ptr[i] << endl;
  }
}

int main(){


  read_act();
  printf("Done reading sparse activations\n");

  read_weights();

  act_val.push_back(0);
  act_ind.push_back(M);
  for(int i=0; i<2*4; ++i) {
    wgt_val.push_back(0); // val
    wgt_val.push_back(M); // col-index
  }
  wgt_ptr[M+1] = wgt_ptr[M]+8;

  printf("Finished reading wgt file\n");

  // print_data();
  
  count_required_multiplications();

  for(int i=0; i<N; ++i) {
    out_vec[i]=0;
  }

  SS_CONFIG(eie_config,eie_size);
  begin_roi();
  mv();
  end_roi();
  sb_stats();

  return 0;
}
