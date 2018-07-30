#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <assert.h>
#include "test.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>

#define VTYPE uint16_t
using namespace std;

// FIXME: padding of 4
void mv_mult(VTYPE *wgt_col_ind, VTYPE *wgt_val, int* wgt_row_ptr, VTYPE *act_ind, VTYPE *act_val, VTYPE *out_vec) {

  int ptr1, end1;
  ptr1=0; end1=0;
  int nnz2 = M*syn_sp;
  nnz2 = (nnz2/4)*4+4;
  
  int ncol = 4;
  SB_CONFIG(test_config,test_size);

  // SB_DMA_WRITE(P_test_out_val, 8, 8, N/4, &out_vec[0]);
  SB_DMA_WRITE(P_test_out_val, 8, 8, ncol/4, &out_vec[0]);
  for (int i=0; i<ncol; ++i){
    ptr1 = wgt_row_ptr[i];
    end1 = wgt_row_ptr[i+1];
	std::cout << "weight row size: " << (end1-ptr1) << " act size: " << nnz2 << "\n";
    
    SB_DMA_READ(&wgt_col_ind[ptr1], 8, 8, (end1-ptr1)/4, P_test_wind);
    SB_DMA_READ(&wgt_val[ptr1], 8, 8, (end1-ptr1)/4, P_test_wval);

	SB_REPEAT_PORT(4);
    SB_DMA_READ(&act_ind[0], 8, 8, nnz2/4, P_test_aind);
	SB_REPEAT_PORT(4);
    SB_DMA_READ(&act_val[0], 8, 8, nnz2/4, P_test_aval);

	// last value in the input should be this sentinal16: these values padded
	// in the input
    // SB_CONST(P_test_wind, SENTINAL, 1);
    // SB_CONST(P_test_aind, SENTINAL, 1);
    // SB_CONST(P_test_wval, 0, 1);
    // SB_CONST(P_test_wind, 0, 1);
  }
  SB_WAIT_ALL(); 
}

int main(){

  char lineToRead[5000];

  VTYPE *act_val;
  VTYPE *act_ind;

  int *wgt_row_ptr;
  VTYPE *wgt_col_ind;
  VTYPE *wgt_val;


  // int nnz = (int)(M*act_sp);
  act_val = (VTYPE*)malloc((int)(M*act_sp+4)*sizeof(VTYPE));
  act_ind = (VTYPE*)malloc((int)(M*act_sp+4)*sizeof(VTYPE));
  int tind=0, tval=0;

  // READING ACTIVATIONS FIRST
  FILE *act_file = fopen("input_activations.data", "r");

  int id=0;
  printf("Start reading activations file\n");

  while(fgets(lineToRead, 5000, act_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> tind >> tval;
	act_ind[id] = (VTYPE)tind;
	act_val[id] = (VTYPE)tval;
	id++;

  }

  // activations are being repeated so here, just push a sentinal at the end
  int rem = 4-(id%4);
  // rem = rem==0?4:rem;
  for(int i=0; i<rem; ++i){
    act_ind[id] = SENTINAL16;
    act_val[id] = 0;
	id++;
  }
  
  printf("Done reading activations file\n");
  fclose(act_file);

  // nnz = (int)(N*M*syn_sp);

  // 4N for extra padding
  wgt_val = (VTYPE*)malloc((int)(N*M*syn_sp+4*N)*sizeof(VTYPE));
  wgt_col_ind = (VTYPE*)malloc((int)(N*M*syn_sp+4*N)*sizeof(VTYPE));
  wgt_row_ptr = (int*)malloc((N+1)*sizeof(VTYPE));

  int row_id; int prev_row_id=-1;
  int len=0;

  // READING WEIGHTS NOW
  FILE *weight_file = fopen("input_weights.data", "r");

  id=0;
  printf("Start reading weights file\n");

  // Empty rows thing won't be a problem here
  while(fgets(lineToRead, 5000, weight_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> row_id >> tind >> tval;
	wgt_col_ind[id] = (VTYPE)tind;
	wgt_val[id] = (VTYPE)tval;

	if(row_id != prev_row_id){
	  if(prev_row_id!=-1)
	    len = id-wgt_row_ptr[prev_row_id];
	  else
		len = id;
	  /*
	  int rem = 4-(len%4);
	  // padding with these values
	  for(int k=0; k<rem; ++k){
		wgt_col_ind[++id]=SENTINAL16;
		wgt_val[id]=0;
	  }
	  */
	  // std::cout << id-wgt_row_ptr[prev_row_id] << "\n";
	  wgt_row_ptr[row_id] = id;
	  prev_row_id = row_id;
	}
	id++;

  }
  // may check it!
  wgt_row_ptr[N] = id;
  
  printf("Done reading weights file\n");
  
  // fclose(weight_file);

  VTYPE *out_vec;
  out_vec = (VTYPE*)malloc(N*sizeof(VTYPE));
  //out_vec = (VTYPE*)malloc(1);
  //
  //
  // MERGING OF 4 ROWS

  
  begin_roi();
  mv_mult(wgt_col_ind, wgt_val, wgt_row_ptr, act_ind, act_val, out_vec);
  end_roi();
  
  return 0;
}
