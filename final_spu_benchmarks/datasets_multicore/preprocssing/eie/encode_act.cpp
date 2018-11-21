#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <inttypes.h>
#include <sstream>
#include <fstream>
#include <queue>

using namespace std;

// fc layer 6
// #define N 9216
// #define M 4096

// fc layer 7
// #define N 4096
// #define M 4096

// very small
#define N 20
#define M 15


// output data structures in RLE format
vector<float> act;
vector<uint16_t> count;

// input data structures 
float dense_act[M];

void encode_rle() {
  int temp_count=0;
  for(int i=0; i<M; ++i) {
	temp_count=0;
	if(dense_act[i]!=0) {
	  act.push_back(dense_act[i]);
	  count.push_back(temp_count);
	  temp_count=0;
	} else {
  	  temp_count++;
	  if(temp_count==64){
	    act.push_back(0);
	    count.push_back(temp_count);
	    temp_count=0;
	  }
	}
  }
}

void store_rle_in_file() {
  ofstream wgt_val ("output_datasets/act_val.data"); // to store nodes in rle
  ofstream wgt_ind ("output_datasets/act_index.data"); // to store nodes in rle

  if(wgt_val.is_open() && wgt_ind.is_open()) {

    for(unsigned k=0; k<act.size(); ++k) {
  	  wgt_val << act[k] << "\n";
  	  wgt_ind << count[k] << "\n";
    }
  }
  wgt_val.close();
  wgt_ind.close();

}

void correctness_check() {
  unsigned nnz=0;
  nnz = act.size();

  float sp = nnz/(float)(1*M);
  cout << "Sparsity percentage is: " << sp << endl;

}

int main() {
  FILE *wgt_file;
  // wgt_file = fopen("input_datasets/fc6_act_file.txt", "r");
  wgt_file = fopen("input_datasets/very_small/act.txt", "r");
  char lineToRead[5000];

  int id=0; 
  int i=0;

  while(fgets(lineToRead, 5000, wgt_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
	float var;

	i = id/M;
	iss >> var;
	if(var!=0) {
	  dense_act[id] = var;
	}
	id++;
  }


  encode_rle();
  correctness_check();
  store_rle_in_file();
  return 0;
}
