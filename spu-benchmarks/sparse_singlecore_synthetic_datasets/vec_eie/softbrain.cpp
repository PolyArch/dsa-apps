#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <assert.h>
#include "vec_test.dfg.h"
#include "../../common/include/ss_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>

#define sentinal (SENTINAL16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48)


#define VTYPE uint16_t
#define vec_width 8
using namespace std;


void mv_merged(uint64_t *wgt_col_ind, uint64_t *wgt_val, int* wgt_row_ptr, uint64_t *act_ind, uint64_t *act_val, uint64_t *out_vec, int act_size) {

  // int ptr1=0, end1=0;
  // int ptr2=0, end2=0;
  // int ptr3=0, end3=0;
  // int ptr4=0, end4=0;
  // int row_size1=0, row_size2=0;
  // int row_size3=0, row_size4=0;
  int ptr[vec_width]; int end[vec_width];
  int row_size[vec_width];

  for(int i=0; i<vec_width; ++i){
	ptr[i]=0; end[i]=0;
	row_size[i]=0;
  }
  
  // int ncol = 1;
  int ncol = N/4;
  // TODO: need to make sure that it is a multiple of vec_width
  SS_CONFIG(vec_test_config,vec_test_size);
  
  SS_DMA_WRITE(P_vec_test_out_val, 8, 8, ncol, &out_vec[0]);
  for (int i=0; i<ncol; i+=vec_width){
  // std::cout << "weight row size: " << row_size << " act size: " << act_size << "\n";
    
    // ROW group 1	
	ptr[0] = wgt_row_ptr[i];
    end[0] = wgt_row_ptr[i+1];
	row_size[0] = (end[0]-ptr[0]);
    SS_DMA_READ(&wgt_col_ind[ptr[0]], 8, 8, row_size[0], P_vec_test_wind0);
	SS_CONST(P_vec_test_wind0, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[0]], 8, 8, row_size[0], P_vec_test_wval0);
	SS_CONST(P_vec_test_wval0, 0, 1);

    // ROW group 2	
	ptr[1] = wgt_row_ptr[i+1];
    end[1] = wgt_row_ptr[i+2];
	row_size[1] = (end[1]-ptr[1]);
    SS_DMA_READ(&wgt_col_ind[ptr[1]], 8, 8, row_size[1], P_vec_test_wind1);
	SS_CONST(P_vec_test_wind1, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[1]], 8, 8, row_size[1], P_vec_test_wval1);
	SS_CONST(P_vec_test_wval1, 0, 1);

    // ROW group 3
	ptr[2] = wgt_row_ptr[i+2];
    end[2] = wgt_row_ptr[i+3];
	row_size[2] = (end[2]-ptr[2]);
    SS_DMA_READ(&wgt_col_ind[ptr[2]], 8, 8, row_size[2], P_vec_test_wind2);
	SS_CONST(P_vec_test_wind2, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[2]], 8, 8, row_size[2], P_vec_test_wval2);
	SS_CONST(P_vec_test_wval2, 0, 1);

	// ROW group 4
    ptr[3] = wgt_row_ptr[i+3];
    end[3] = wgt_row_ptr[i+4];
	row_size[3] = (end[3]-ptr[3]);
    SS_DMA_READ(&wgt_col_ind[ptr[3]], 8, 8, row_size[3], P_vec_test_wind3);
	SS_CONST(P_vec_test_wind3, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[3]], 8, 8, row_size[3], P_vec_test_wval3);
	SS_CONST(P_vec_test_wval3, 0, 1);


	// ROW group 5
	ptr[4] = wgt_row_ptr[i+4];
    end[4] = wgt_row_ptr[i+5];
	row_size[4] = (end[4]-ptr[4]);
    SS_DMA_READ(&wgt_col_ind[ptr[4]], 8, 8, row_size[4], P_vec_test_wind4);
	SS_CONST(P_vec_test_wind4, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[4]], 8, 8, row_size[4], P_vec_test_wval4);
	SS_CONST(P_vec_test_wval4, 0, 1);

    // ROW group 2	
	ptr[5] = wgt_row_ptr[i+5];
    end[5] = wgt_row_ptr[i+6];
	row_size[5] = (end[5]-ptr[5]);
    SS_DMA_READ(&wgt_col_ind[ptr[5]], 8, 8, row_size[5], P_vec_test_wind5);
	SS_CONST(P_vec_test_wind5, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[5]], 8, 8, row_size[5], P_vec_test_wval5);
	SS_CONST(P_vec_test_wval5, 0, 1);

    // ROW group 3
	ptr[6] = wgt_row_ptr[i+6];
    end[6] = wgt_row_ptr[i+7];
	row_size[6] = (end[6]-ptr[6]);
    SS_DMA_READ(&wgt_col_ind[ptr[6]], 8, 8, row_size[6], P_vec_test_wind6);
	SS_CONST(P_vec_test_wind6, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[6]], 8, 8, row_size[6], P_vec_test_wval6);
	SS_CONST(P_vec_test_wval6, 0, 1);

	// ROW group 4
    ptr[7] = wgt_row_ptr[i+7];
    end[7] = wgt_row_ptr[i+8];
	row_size[7] = (end[7]-ptr[7]);
    SS_DMA_READ(&wgt_col_ind[ptr[7]], 8, 8, row_size[7], P_vec_test_wind7);
	SS_CONST(P_vec_test_wind7, sentinal, 1);
    SS_DMA_READ(&wgt_val[ptr[7]], 8, 8, row_size[7], P_vec_test_wval7);
	SS_CONST(P_vec_test_wval7, 0, 1);

	SS_REPEAT_PORT(4);
    SS_DMA_READ(&act_ind[0], 8, 8, act_size, P_vec_test_aind);
	SS_REPEAT_PORT(4);
    SS_DMA_READ(&act_val[0], 8, 8, act_size, P_vec_test_aval);
  }
  SS_WAIT_ALL(); 
}


int main(){

  char lineToRead[5000];

  VTYPE *act_val;
  VTYPE *act_ind;

  int *wgt_row_ptr;
  VTYPE *wgt_col_ind;
  VTYPE *wgt_val;


  int nnz = (int)(M*act_sp);
  // make sure this is of the form 4k+3 (break after id is equal to that)
  int to_read_values = ((nnz-3)/4)*4+3;
  int act_size = to_read_values+1;

  act_val = (VTYPE*)malloc(act_size*sizeof(VTYPE));
  act_ind = (VTYPE*)malloc(act_size*sizeof(VTYPE));
  int tind=0, tval=0;

  // READING ACTIVATIONS FIRST
  FILE *act_file = fopen("input_activations.data", "r");

  int id=0;
  // printf("Start reading activations file\n");

  while(fgets(lineToRead, 5000, act_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> tind >> tval;
	act_ind[id] = (VTYPE)tind;
	act_val[id] = (VTYPE)tval;
	if(id==(to_read_values-1)){
	  break;
	}
	id++;

  }

  act_ind[to_read_values] = SENTINAL16;
  act_val[to_read_values] = 0;
  // activations are being repeated so here, just push a sentinal at the end
  uint64_t *act_merged_val;
  uint64_t *act_merged_ind;
  int id3=0;
  int merged_act_size = act_size/4;
  act_merged_ind = (uint64_t*)malloc(merged_act_size*sizeof(uint64_t));
  act_merged_val = (uint64_t*)malloc(merged_act_size*sizeof(uint64_t));

  VTYPE temp_val[4]; VTYPE temp_id[4];
  
  int count=0;
  for(int i=0; i<act_size; ++i){
	// temp_id[count] = act_val[i];
	// temp_val[count] = act_val[i];
    count++;

	if(count==4){
	  act_merged_ind[id3] = (act_ind[i] | act_ind[i-1] << 16 | (act_ind[i-2] & 0xFFFFFFFFFFFFFFFF) << 32 | (act_ind[i-3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  act_merged_val[id3] = (act_val[i] | act_val[i-1] << 16 | (act_val[i-2] & 0xFFFFFFFFFFFFFFFF) << 32 | (act_val[i-3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  id3++;
	  count=0;
	}
  }


  // printf("Done reading activations file\n");
  fclose(act_file);
  free(act_val);
  free(act_ind);

  // nnz = (int)(N*M*syn_sp);

  // 4N for extra padding
  wgt_val = (VTYPE*)malloc((int)(N*M*syn_sp+4*N)*sizeof(VTYPE));
  wgt_col_ind = (VTYPE*)malloc((int)(N*M*syn_sp+4*N)*sizeof(VTYPE));
  wgt_row_ptr = (int*)malloc((N+1)*sizeof(int));

  int row_id; int prev_row_id=-1;
  int len=0;

  // READING WEIGHTS NOW
  FILE *weight_file = fopen("input_weights.data", "r");

  id=0;
  // printf("Start reading weights file\n");

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
	  
	  // std::cout << id-wgt_row_ptr[prev_row_id] << "\n";
	  wgt_row_ptr[row_id] = id;
	  prev_row_id = row_id;
	}
	id++;
  }
  // may check it!
  wgt_row_ptr[N] = id;
  
  // printf("Done reading weights file\n");
  
  fclose(weight_file);

  // MERGING OF 4 ROWS
  uint64_t *wgt_merged_col_ind;
  uint64_t *wgt_merged_val;
  int *wgt_merged_row_ptr;
  int syn_size = N*(M*syn_sp/4+1);
  wgt_merged_col_ind = (uint64_t*)malloc(syn_size*4*sizeof(uint64_t));
  wgt_merged_val = (uint64_t*)malloc(syn_size*4*sizeof(uint64_t));
  wgt_merged_row_ptr = (int*)malloc((int(N/4)+1)*4*sizeof(int));

  int id2=0;
  int counter=0;
  int offset=0;
  // printf("Entering the loop\n");
  for(int j=0; j<N/4; ++j){
	offset = j*4*M*syn_sp;
	// NEED TO MERGE 4 ROWS AT A TIME
    for(int i=0; i<(M*syn_sp); ++i){
	  wgt_merged_val[id2] = (wgt_val[offset+i] | wgt_val[offset+i+1] << 16 | (wgt_val[offset+i+2] & 0xFFFFFFFFFFFFFFFF) << 32 | (wgt_val[offset+i+3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  wgt_merged_col_ind[id2] = (wgt_col_ind[offset+i] | wgt_col_ind[offset+i+1] << 16 | (wgt_col_ind[offset+i+2] & 0xFFFFFFFFFFFFFFFF) << 32 | (wgt_col_ind[offset+i+3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  id2++;
    }

    // wgt_merged_val[id2] = 0;
    // wgt_merged_col_ind[id2] = (SENTINAL16 | SENTINAL16 << 16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48);
	// id2++;
	wgt_merged_row_ptr[j] = wgt_row_ptr[j*4]/4;
	// std::cout << "id2: " << id2 << "\n";
  }
  wgt_merged_row_ptr[N/4] = syn_size;

  free(wgt_val);
  free(wgt_col_ind);
  free(wgt_row_ptr);
  printf("Done writing merged weights values\n");
  /*
  for(int i=0; i<id2; ++i){
	std::cout << "ind: " << std::hex << wgt_merged_col_ind[i] << "\n";
	std::cout << "val: " << std::hex << wgt_merged_val[i] << "\n";
  }
  */
  uint64_t *out_vec;
  out_vec = (uint64_t*)malloc(N/4*sizeof(uint64_t));

  /*
  for(int i=0; i<N/4; ++i){
	std::cout << "CAME HERE\n";
    std::cout << std::hex << &out_vec[i] << "\n";
  }
  */

  // mv_merged(wgt_merged_col_ind, wgt_merged_val, wgt_merged_row_ptr, act_ind, act_val, out_vec, act_size);
  begin_roi();
  mv_merged(wgt_merged_col_ind, wgt_merged_val, wgt_merged_row_ptr, act_merged_ind, act_merged_val, out_vec, merged_act_size);
  end_roi();
  sb_stats();
  
  // begin_roi();
  // mv_mult(wgt_col_ind, wgt_val, wgt_row_ptr, act_ind, act_val, out_vec);
  // end_roi();
  
  return 0;
}
