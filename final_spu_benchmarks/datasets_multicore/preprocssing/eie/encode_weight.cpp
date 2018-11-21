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
vector<float> wgt[N];
vector<uint16_t> count[N];

// input data structures 
float dense_wgt[N][M];

void encode_rle() {
  int temp_count=0;
  for(int i=0; i<N; ++i) {
	temp_count=0;
	for(int j=0; j<M; ++j) {
	  if(dense_wgt[i][j]!=0){
		wgt[i].push_back(dense_wgt[i][j]);
		count[i].push_back(temp_count);
		temp_count=0;
	  } else {
		temp_count++;
		if(temp_count==64){
		  wgt[i].push_back(0);
		  count[i].push_back(temp_count);
		  temp_count=0;
		}
	  }
	}
  }
}

void store_rle_in_file() {
  ofstream wgt_val ("output_datasets/wgt_val.data"); // to store nodes in rle
  ofstream wgt_ind ("output_datasets/wgt_index.data"); // to store nodes in rle

  if(wgt_val.is_open() && wgt_ind.is_open()) {

	for(int i=0; i<N; i++) {
	  for(unsigned k=0; k<wgt[i].size(); ++k) {
		wgt_val << wgt[i][k] << "\n";
		wgt_ind << count[i][k] << "\n";
	  }
    }
  }
  wgt_val.close();
  wgt_ind.close();

  ofstream wgt_ptr ("output_datasets/wgt_ptr.data"); // to store nodes in rle

  if(wgt_ptr.is_open()) {

	unsigned cum_size=0;
	for(int i=0; i<N; i++) {
	  cum_size += count[i].size();
	  wgt_ptr << cum_size << "\n";
    }
  }
  wgt_ptr.close();


}

void correctness_check() {
  unsigned nnz=0;
  for(int i=0; i<N; ++i) {
	nnz += wgt[i].size();
  }

  float sp = nnz/(float)(N*M);
  cout << "Sparsity percentage is: " << sp << endl;

}

int main() {
  FILE *wgt_file;
  // wgt_file = fopen("input_datasets/fc_weight_file.txt", "r");
  // wgt_file = fopen("input_datasets/fc7_wgt_file.txt", "r");
  // wgt_file = fopen("input_datasets/pytorch_fc6_wgt.txt", "r");
  wgt_file = fopen("input_datasets/very_small/wgt.txt", "r");
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
	  dense_wgt[i][(id%M)] = var;
	  // wgt[i].push_back(var);
	  // count[i].push_back(id%M);
	}
	id++;
  }


  encode_rle();
  correctness_check();
  store_rle_in_file();
  return 0;
}
