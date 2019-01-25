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
#include <cstring>
#include <assert.h>

using namespace std;

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
		  // if(dense_weights[k][i][n]!=0) {
		   if(dense_weights[k][i][n]>0.1 || dense_weights[k][i][n]<-0.1) {
		        synapse[i][j/Tn].push_back(dense_weights[k][i][n]);
			      count[i][j/Tn].push_back(temp_count+1);
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
        s += synapse[i][j].size();
        /*
	    for(unsigned k=0; k<count[i][j].size(); ++k) {
		    s += count[i][j][k];
	    }
        */
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
  float sp = s/(float)(Kx*Ky*Nn*Ni);
  cout << "Sparsity %age is: " << sp << endl;
}

void store_rle_in_file() {
  string str(net_name);
  char x1[100] = "output_datasets/";
  char x2[100] = "output_datasets/";
  char x3[100] = "output_datasets/";
  char a[100] = "/wgt_val.data";
  char b[100] = "/wgt_index.data";
  char c[100] = "/wgt_ptr.data";


  ofstream syn_val (strcat(strcat(x1,str.c_str()),a));
  ofstream syn_ind (strcat(strcat(x2,str.c_str()),b));
  // ofstream syn_val ("output_datasets/wgt_val.data"); // to store nodes in rle
  // ofstream syn_ind ("output_datasets/wgt_index.data"); // to store nodes in rle

  if(syn_val.is_open() && syn_ind.is_open()) {

	/* for(int i=0; i<Ni; i++)*/ int i=0; {
	  /* for(int j=0; j<Nn; j+=Tn)*/ int j=0; {
		for(unsigned k=0; k<synapse[i][j/Tn].size(); ++k) {
		  syn_val << synapse[i][j/Tn][k] << "\n";
		  syn_ind << count[i][j/Tn][k] << "\n";
		}
	  }
    }
  }
  syn_val.close();
  syn_ind.close();

  ofstream syn_ptr (strcat(strcat(x3,str.c_str()),c));
  // ofstream syn_ptr ("output_datasets/wgt_ptr.data"); // to store nodes in rle
  unsigned cum_size=0;

  if(syn_ptr.is_open()) {

	/* for(int i=0; i<Ni; i++) */ int i=0; {
	  /*for(int j=0; j<Nn/Tn; j++)*/ int j=0; {
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
  // syn_file = fopen("input_datasets/very_small/wgt.txt", "r");
  string str(net_name);
  char a[100] = "input_datasets/";
  char b[100] = "/wgt.txt";
  // act_file = fopen(strcat(str.c_str(),"act_conv.data"));
  syn_file = fopen(strcat(strcat(a,str.c_str()),b), "r");
 
  char lineToRead[5000];

  int first;
  int second;
  int third;
  int nnz=0;

  while(fgets(lineToRead, 5000, syn_file) != NULL) {
	std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
	float val;

    iss >> val;
	first = nnz%(Kx*Ky);
	// second = (nnz%(Ni*Kx*Ky))/(Ni);
	second = (nnz/(Kx*Ky))%Ni;
    third = nnz/(Ni*Kx*Ky);
    // Kx*Ky, Ni, Nn
    assert(second<Ni);
    assert(third<Nn);
	dense_weights[first][second][third] = val;

    nnz++;
  }

  fclose(syn_file);
  cout << "Done reading weights\n";

  reformat_layout();
  correctness_check(nnz);
  store_rle_in_file();

  return 0;
}
