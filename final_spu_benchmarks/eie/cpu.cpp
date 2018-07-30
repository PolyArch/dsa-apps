#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <inttypes.h>
#include <assert.h>
#include <sstream>
#include "../../common/include/sim_timing.h"

// #define VTYPE typeof(int)
#define VTYPE uint16_t
// using namespace std;

void mv_mult(VTYPE *wgt_col_ind, VTYPE *wgt_val, int* wgt_row_ptr, VTYPE *act_ind, VTYPE *act_val, VTYPE *out_vec) {

  // std::cout << "chalo yahan tak to aaya\n";

  /*
  for(int i=0; i<=N; ++i){
	std::cout << "i = "<< i <<" in for loop "<<wgt_row_ptr[i] << "\n";
  }
  */

  int ptr1, ptr2, end1, end2;
  ptr1=0; end1=0;
  ptr2 = 0; 
  end2 = (int)(M*syn_sp);


  VTYPE accum = 0;

  for (int i=0; i<N; i++){

    ptr1 = wgt_row_ptr[i];
    end1 = wgt_row_ptr[i+1];
	// std::cout << ptr1 << " " << end1 << "\n";
    // make sure that we don't need it: in synthetic we won't need
    // if(ptr1 == -1 || ptr2 == -1)
    //   continue;
    accum = 0;

    while(ptr1 <= end1 && ptr2 <= end2){
      if(wgt_col_ind[ptr1] == act_ind[ptr2]){
        accum += (VTYPE)(wgt_val[ptr1]*act_val[ptr2]);
        ptr1++; ptr2++;
      }
      else{
        if(wgt_col_ind[ptr1] <= act_ind[ptr2])
          ptr1++;
        else
          ptr2++;
      }
    }
    out_vec[i] = (VTYPE)accum;
  }
  /*
  printf("printing the output non-zero values\n");
  for (int i=0; i<=last; i++){
      printf("%ld\n",vector3[i].second);
  }
  */
}

int main(){

  char lineToRead[5000];

  VTYPE *act_val;
  VTYPE *act_ind;

  int *wgt_row_ptr;
  VTYPE *wgt_col_ind;
  VTYPE *wgt_val;


  // int nnz = (int)(M*act_sp);
  act_val = (VTYPE*)malloc((int)(M*act_sp)*sizeof(VTYPE));
  act_ind = (VTYPE*)malloc((int)(M*act_sp)*sizeof(VTYPE));
  int tind=0, tval=0;

  // READING ACTIVATIONS FIRST
  FILE *act_file = fopen("input_activations.data", "r");

  int id=0;
  printf("Start reading activations file\n");

  while(fgets(lineToRead, 5000, act_file) != NULL){
	// std::cout << "Hi\n";
	// std::cout << lineToRead << "\n";
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> tind >> tval;
	act_ind[id] = (VTYPE)tind;
	// act_ind[id] = (uint16_t)tind;
	act_val[id] = (VTYPE)tval;
	// iss >> act_ind[id] >> act_val[id];
	id++;
	// std::cout << "Hi2\n";

  }
  
  printf("Done reading activations file\n");
  fclose(act_file);

  // nnz = (int)(N*M*syn_sp);

  wgt_val = (VTYPE*)malloc((int)(N*M*syn_sp)*sizeof(VTYPE));
  wgt_col_ind = (VTYPE*)malloc((int)(N*M*syn_sp)*sizeof(VTYPE));
  // wgt_row_ptr = (int*)calloc((N+1)*sizeof(VTYPE),0);
  wgt_row_ptr = (int*)malloc((N+1)*sizeof(VTYPE));

  int row_id; int prev_row_id=-1;

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
	// iss >> row_id >> wgt_col_ind[id] >> wgt_val[id];
	// std::cout << "id: " << id << " " << row_id << " " << wgt_col_ind[id] << " " << wgt_val[id] << "\n";

	if(row_id != prev_row_id){
	//std::cout << "before id = " << id << "index =  " << row_id << " wgt[] = " << wgt_row_ptr[row_id] << "\n";
	  wgt_row_ptr[row_id] = id;
	//std::cout << "after id = " << id << "index =  " << row_id << " wgt[] = " << wgt_row_ptr[row_id] << "\n";
	  prev_row_id = row_id;
	}
	id++;

  // std::cout << " val 504=  " << " wgt[] = " << wgt_row_ptr[504] << "\n";

  }
  // may check it!
  wgt_row_ptr[N] = id;
//std::cout << "last val N id = " << id << "N =  " << N << " wgt[] = " << wgt_row_ptr[N] << "\n";
  
  printf("Done reading weights file\n");
  /*
  for(int i=0; i<=N; ++i){
	std::cout << "i = "<< i <<" in for loop "<<wgt_row_ptr[i] << "\n";
  }
  */
  // fclose(weight_file);

  //FIXME
  VTYPE *out_vec;
  out_vec = (VTYPE*)malloc(N*sizeof(VTYPE));
  //out_vec = (VTYPE*)malloc(1);

  // int *out_vec2;
  // out_vec = (VTYPE*)malloc(N*sizeof(VTYPE));
  // out_vec2 = (int*)malloc(1*sizeof(int));
  
  // begin_roi();
  mv_mult(wgt_col_ind, wgt_val, wgt_row_ptr, act_ind, act_val, out_vec);
  // end_roi();
  
  return 0;
}
