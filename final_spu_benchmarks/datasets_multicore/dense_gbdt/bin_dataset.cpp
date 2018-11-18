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
#include <algorithm>
#include <functional>
#include <numeric>

using namespace std;

#define N 100
#define M 136
#define K 64

// output data datastructures
int bin_inst_feat[N][M];

// input data structures
float inst_feat[N][M];
int inst_label[N];

float min_val[M] = {10000};
float max_val[M] = {-10000};

float quantile[N][M];

int final_count = N; // bin each feature seperately.

// each bin in Histogram should have data of fixed quantile
/*
void get_quantile() {
   for(int i=0; i<M; ++i){
     vector<float> local_inst;
     for(int j=0; j<N; ++j){
       local_inst.push_back(inst_feat[j][i]);
     }
	 vector<int> idx(local_inst.size());
	 iota(idx.begin(), idx.end(), 0);

	 // sort indexes based on comparing values in v
	 sort(idx.begin(), idx.end(),
		  [&local_inst](int i1, int i2) {return local_inst[i1] < local_inst[i2];});
	 // sort(local_inst.begin(), local_inst.end());
     // now we know the percentile

   }



}
*/

// TODO: make it more efficient (maybe using python)
int get_rank(float val, int feat_id) {
  int count=0;
  for(int i=0; i<N; ++i){
	if(val < inst_feat[i][feat_id]){
	  count++;
	}
  }
  return count;
}

void get_quantile() {
  // cout << "Come inside the quantile function\n";
  for(int i=0; i<N; ++i){
    for(int j=0; j<M; ++j){
      // now we know the percentile
      quantile[i][j] = float(get_rank(inst_feat[i][j], j))/(float)N;
      bin_inst_feat[i][j] = quantile[i][j]*K;
	  // cout << "binned value: " << bin_inst_feat[i][j] << endl;
    }
	// cout << "Complete 1 feature\n";
  }
}

// set min and max for all features
void set_min_max() {
  for(int i=0; i<N; i++){
    for(int j=0; j<M; ++j){
      max_val[j] = max(inst_feat[i][j], max_val[j]);
      min_val[j] = min(inst_feat[i][j], min_val[j]);
      /*
      if(inst_feat[i][j]<min_val[j]){
        min_val[j] = inst_feat[i][j];
      }
      if(inst_feat[i][j]>max_val[j]){
        max_val[j] = inst_feat[i][j];
      }
      */
    }
  }

  for(int i=0; i<M; ++i){
    cout << min_val[i] << " " << max_val[i] << endl;
  }
}

void bin_dataset() {
  float a, b, c;
  for(int i=0; i<N; i++){
    for(int j=0; j<M; ++j){
      // FIXME: set the float things
      a = (inst_feat[i][j]-min_val[j])*K;
      b = max_val[j]-min_val[j];
      c = a/b;
      bin_inst_feat[i][j] = (int)(c);
    }
  }
  set_min_max();
}

void store_data_in_file() {
  ofstream data_file ("binned_small_mslr.train"); // to store the nodes in top_order

  // write to the file here
  if(data_file.is_open()) {
    for(int i=0; i<N; i++){
      data_file << inst_label[i] << " ";
      for(int j=0; j<M; ++j){
        data_file << bin_inst_feat[i][j] << " "; // let's reduce the size of the file
      }
      data_file << "\n";
    }
  }
  data_file.close();
}


int main() {
  FILE *data;
  data = fopen("small_mslr.train", "r");
  char lineToRead[5000];
  int n=0;

  while (fgets(lineToRead, 5000, data) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    char op;
    int ignore;

    iss >> inst_label[n];
    for(int i=0; i<M; i++){
  	  iss >> ignore >> op >> inst_feat[n][i];
      // cout << inst_feat[n][i] << " ";
  	}
    // cout << "\n";
  	n++;
  }

  fclose(data);
  cout << "Done reading the input file\n";

  // bin_dataset();
  get_quantile();
  store_data_in_file();
  cout << "Done writing the binned file\n";

  return 0;
}
