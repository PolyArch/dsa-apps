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

#define SENTINAL (uint32_t)(1<<31)

using namespace std;

// small adult
// #define N 84
// #define M 100
// #define M 39056

// diabetes
#define N 8
#define M 768

// input data structures
// double py[M];
// vector<float> pdata_val[M];
// vector<uint32_t> pdata_ind[M];


// output data structures
double y[M];
vector<float> data_val[M];
vector<uint32_t> data_ind[M];

void pad_dataset() {
  for(unsigned i=0; i<M; ++i) {
	if(data_val[i].size()%2!=0){
	  data_val[i].push_back(0);
	  data_ind[i].push_back(SENTINAL);
	} else {
      for(int i=0; i<2; ++i) {
        data_val[i].push_back(0);
	    data_ind[i].push_back(SENTINAL);
      }
    }
  }
}

void store_padded_in_file() {
  // ofstream data_file ("output_datasets/small_adult.data");
  ofstream data_file ("output_datasets/diabetes.data"); // to store nodes in csr

  for(int i=0; i<M; ++i) {
	data_file << y[i] << " ";
	for(unsigned j=0; j<data_val[i].size(); ++j) {
	  data_file << data_ind[i][j] << ":" << data_val[i][j] << " ";
	}
	data_file << "\n";
  }

  data_file.close();
}

int main() {
  FILE *ksvm_file;
  // ksvm_file = fopen("input_datasets/small_adult.data", "r");
  ksvm_file = fopen("input_datasets/diabetes.data", "r");
  char lineToRead[5000];

  int inst_id=0;
  while(fgets(lineToRead, 5000, ksvm_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
	char ignore;
	float x;
	int ind;

	iss >> y[inst_id];

	while(iss >> ind) {
	  iss >> ignore >> x;
	  data_ind[inst_id].push_back(ind);
	  data_val[inst_id].push_back(x);
	}
	
    inst_id++;;

  }
  fclose(ksvm_file);

  pad_dataset();
  store_padded_in_file();
  return 0;
}
