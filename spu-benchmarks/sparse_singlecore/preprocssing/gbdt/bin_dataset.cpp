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
#include <cstring>

using namespace std;

// #define N 192156
// #define N 50
// #define M 136

// #define N 1056
// #define M 126

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
      // FIXME:HACK: to maintian the sparsity
      
      /*
      if(inst_feat[i][j]==0) {
        bin_inst_feat[i][j]=0;
      } else {
        */
        quantile[i][j] = float(get_rank(inst_feat[i][j], j))/(float)N;
        bin_inst_feat[i][j] = quantile[i][j]*K;
     // }
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

// need output in sparse format
void store_data_in_file() {
  string str2(dataset);
  char a2[100] = "output_datasets/";
 
  // ofstream data_file ("binned_small_mslr.train");
  ofstream data_file (strcat(a2,str2.c_str()));
  int nnz=0;

  // write to the file here
  if(data_file.is_open()) {
    for(int i=0; i<N; i++){
      data_file << inst_label[i] << " ";
      // for(int j=0; j<M; ++j){
      for(int j=0; j<feat_needed; ++j){
        if(bin_inst_feat[i][j]!=0) { 
          nnz++;
          data_file << j << ":" << bin_inst_feat[i][j] << " "; // let's reduce the size of the file
        }
      }
      data_file << "\n";
    }
  }
  data_file.close();

  float sp = nnz/(float)(N*M);
  cout << "Sparsity in the gbdt dataset is: " << sp <<endl;
}


int main() {


  // initialization
  for(int i=0; i<N; ++i) {
    for(int j=0; j<M; ++j) {
      inst_feat[i][j]=0;
    }
  }

  FILE *data;
  // data = fopen("datasets/small_mslr.train", "r");
  // data = fopen("datasets/connect-4.data", "r");
  // data = fopen("datasets/small_yahoo.train", "r");
  string str1(dataset);
  char a1[100] = "datasets/";
  data = fopen(strcat(a1,str1.c_str()), "r");
  char lineToRead[5000];
  int n=0;
  int nnz=0;

  cout << "Started reading the input file\n";
  while (fgets(lineToRead, 5000, data) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    char op;
    int ignore;
    int var;
    int fid;

    iss >> inst_label[n];
    /*
    for(int i=0; i<M; i++){
      iss >> fid >> op >> var;
      inst_feat[n][fid] = var;
      if(inst_feat[n][fid]!=0) nnz++;
    }
    */
    
    while(iss>>fid) {
  	  iss >> op >> inst_feat[n][fid];
      if(inst_feat[n][fid]!=0) nnz++;
    // cout << "n:" << n << " fid:" << fid<< "\n";
  	}
    if(n==N) break;
    // cout << "n:" << n << " nnz:" << nnz<< " ";
    
    // cout << "\n";
  	n++;
  }

  fclose(data);
  cout << "Done reading the input file\n";
  float sp = nnz/(float)(N*M);
  cout << "Initial sparsity is: " << sp << endl;

  // bin_dataset();
  get_quantile();
  store_data_in_file();
  cout << "Done writing the binned file\n";

  return 0;
}
