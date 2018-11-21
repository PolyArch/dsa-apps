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

// input to conv layer 1
// #define Ni 3
// #define Nn 96
// #define Tn 96
// #define Kx 11
// #define Ky 11
// #define stride 4

// input to conv layer 2
// #define Ni 96
// #define Nn 256
// #define Tn 64
// #define Kx 5
// #define Ky 5
// #define stride 2

// input to very small
#define Ni 1
#define Nn 1
#define Tn 1
#define Kx 3
#define Ky 3

// output data structures in RLE format
vector<float> synapse[Ni][Nn/Tn]; // other side is nnz
vector<int> count[Ni][Nn/Tn]; // other side is nnz

// input data structures
// vector<int> out_dim_ind;
// vector<int> other_dim_ind;
// vector<float> weights;

float dense_weights[Kx*Ky][Ni][Nn];

void reformat_layout() {
  int stride = Kx*Ky;
  cout << "stride: " << stride << endl;
  int temp_count = 0;
  // Let's do rle here only
  for(int i=0; i<Ni; i++) {
	  for(int j=0; j<Nn; j+=Tn) {
      for(int n=j; n<(j+Tn) && n<Nn; ++n) {
        for(int k=0; k<Kx*Ky; ++k) {
          // cout << "k: " << k << " i: " << i << " n: " << n << endl;
		      if(dense_weights[k][i][n]!=0) {
		        synapse[i][j/Tn].push_back(dense_weights[k][i][n]);
            // cout << "temp_count: " << temp_count << endl;
			      count[i][j/Tn].push_back(temp_count);
			      temp_count=0;
		      } else {
			      temp_count++;
			      // insert 0s won't be an issue for 16-bit values
			      if(temp_count==64) {
			        synapse[i][j/Tn].push_back(0);
			        count[i][j/Tn].push_back(temp_count);
			        temp_count=0;
			      }
		      }
	      }
	    }
	  }
  }
}

void correctness_check(int nnz) {
  int s=0;
  for(int i=0; i<Ni; ++i) {
	  for(int j=0; j<Nn/Tn; ++j) {
	    for(unsigned k=0; k<count[i][j].size(); ++k) {
		    s += count[i][j][k];
	    }
	  }
  }
  cout << "Number of zero values: " << s << endl;
  int count_nnz = (Kx*Ky*Ni*Nn)-s;
  if(count_nnz==nnz){
    cout << "CORRECT!" << endl;
  } else {
    cout << "ERROR!" << endl;
    cout << "Number of non-zero values: " << count_nnz << " and expected: " << nnz << endl;
  }
}

void store_rle_in_file() {
  ofstream syn_val ("output_datasets/wgt_val.data"); // to store nodes in rle
  ofstream syn_ind ("output_datasets/wgt_index.data"); // to store nodes in rle

  if(syn_val.is_open() && syn_ind.is_open()) {

	for(int i=0; i<Ni; i++) {
	  for(int j=0; j<Nn; j+=Tn) {
		for(unsigned k=0; k<synapse[i][j/Tn].size(); ++k) {
		  syn_val << synapse[i][j/Tn][k] << "\n";
		  syn_ind << count[i][j/Tn][k] << "\n";
		}
	  }
    }
  }
  syn_val.close();
  syn_ind.close();

  ofstream syn_ptr ("output_datasets/wgt_ptr.data"); // to store nodes in rle
  unsigned cum_size=0;

  if(syn_ptr.is_open()) {

	for(int i=0; i<Ni; i++) {
	  for(int j=0; j<Nn/Tn; j++) {
		cum_size += synapse[i][j].size();
		syn_ptr << cum_size << "\n";
	  }
    }
  }
  syn_ptr.close();

}

void initialize_weights() {
  for(int i=0; i<Kx*Ky; ++i) {
	  for(int j=0; j<Ni; ++j) {
	    for(int k=0; k<Nn; ++k) {
		    dense_weights[i][j][k] = 0.0f;
	    }
	  }
  }
}

int main() {

  initialize_weights();

  FILE *syn_file;
  // syn_file = fopen("input_datasets/alex_conv2.mtx", "r");
  syn_file = fopen("input_datasets/very_small/wgt.txt", "r");
  char lineToRead[5000];

  bool first_time = true;
  int nnz;
  // int i=0;
  while(fgets(lineToRead, 5000, syn_file) != NULL) {
	std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    // char op;
	  float val;
    int out_dim;
    int other_dim;

/*
	iss >> op;

	if(op == '%'){
	  continue;
	}
  */
	if(first_time) {
	  iss >> out_dim >> other_dim >> nnz;
    // cout << out_dim << " " << other_dim << " " << nnz << endl;
	  first_time = false;
	  // out_dim_ind.resize(nnz);
	  // other_dim_ind.resize(nnz);
	  // weights.resize(nnz);
	} else {
	  iss >> out_dim >> other_dim >> val;

    // just because of the format they use
    out_dim--;
    other_dim--;

	  int first = other_dim%(Kx*Ky);
	  int second = other_dim/(Kx*Ky);
    // cout << first << " " << second << " " << out_dim << endl;
	  dense_weights[first][second][out_dim] = val;

	  // out_dim_ind[i] = out_dim;
	  // other_dim_ind[i] = out_dim;
	  // weights[i] = out_dim;
	  // i++;
	}
  }

  fclose(syn_file);

  reformat_layout();
  correctness_check(nnz);
  store_rle_in_file();

  return 0;
}
