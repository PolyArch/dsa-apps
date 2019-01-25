#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include "test.dfg.h"
#include "../../common/include/ss_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>
#include <sstream>

using namespace std;

#define fused_const (1 | 1 << 16 | (1 & 0xFFFFFFFFFFFFFFFF) << 32 | (1 & 0xFFFFFFFFFFFFFFFF) << 48)

union a{
  float x;
  int16_t y;
};

int main() {

  int num_inputs = Tx*Tx*Tn;
  FILE *out_neuron_file = fopen("output_neuron.data","r"); 
  int t=0; float t2;
  union a temp;
  int16_t out_n[num_inputs];
  char lineToRead[5000];
  printf("Started reading file!\n");
  while(fgets(lineToRead, 5000, out_neuron_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	for(int i=0; i<Tx*Tx; ++i){
	  iss >> temp.x;
	  // iss >> t2;
	  // out_n[t*Tn+i] = (int16_t)(t2 * (1 << 8));
	  out_n[t*Tn+i] = temp.y;
	  // cout << temp.x << " " << out_n[i] << endl;
	  // cout << t2 << " " << out_n[i] << endl;
	  // iss >> out_n[t*Tn+i];
	}
	t++;
  }

  out_n[num_inputs-1] = SENTINAL16;
  cout << out_n[num_inputs-1] << endl;

  printf("Done reading file!\n");
  /*
  uint64_t x[num_inputs/4];
  for(uint64_t i=0; i<num_inputs/4; ++i){
	x[i]=i+2;
  }
  */

  // SS_SCRATCH_DMA_STORE(0, 8, 8, num_inputs/4, &out_n[0]);
  // SS_SCRATCH_DMA_STORE(0, 8, 8, num_inputs/4, &x[0]);
  // SS_WAIT_SCR_WR();

  uint16_t n_val[num_inputs];
  uint16_t n_ind[num_inputs];
  begin_roi();
  SS_CONFIG(test_config, test_size);

  SS_REPEAT_PORT(4);
  // SS_SCRATCH_READ(0, (num_inputs*2), P_test_neuron);
  SS_DMA_READ(&out_n[0], 8, 8, num_inputs/4, P_test_neuron);
  SS_CONST(P_test_const, fused_const, num_inputs);
  SS_CONST(P_test_dummy, 1, num_inputs);
  SS_DMA_WRITE_SIMP(P_test_nval, num_inputs*2, &n_val[0]);
  SS_DMA_WRITE_SIMP(P_test_nind, num_inputs*2, &n_ind[0]);
  SS_RECV(P_test_done, temp);
  SS_RESET();
  SS_WAIT_ALL();
  end_roi();
  sb_stats();
  return 0;
}
