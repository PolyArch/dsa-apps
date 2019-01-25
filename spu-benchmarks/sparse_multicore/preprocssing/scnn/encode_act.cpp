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
#include <string.h>

using namespace std;

// input to conv layer 1
// #define Ni 3
// #define Nx 227
// #define Ny 227

// input to conv layer 2
// #define Ni 96
// #define Nx 55
// #define Ny 55
// #define Tx 7
// #define Ty 7

// input to very small
// #define Ni 1
// #define Nx 10
// #define Ny 10
// #define Tx 10
// #define Ty 10


// output data structures in RLE format
const int dim1 = (Nx*Ny)/(Tx*Ty);
vector<float> act[dim1][Ni];
vector<float> count[(Nx*Ny)/(Tx*Ty)][Ni];

// input data structures 
float dense_act[Nx][Ny][Ni];

string str;

void encode_rle() {

  int temp_count=0;
  int x_dim=0;

  for(int i=0; i<Nx; ++i) {
	for(int j=0; j<Ny; ++j) {
	  for(int k=0; k<Ni; ++k) {
		x_dim = (i*j)/(Tx*Ty);

		if(dense_act[i][j][k]!=0){
		  act[x_dim][k].push_back(dense_act[i][j][k]);
		  count[x_dim][k].push_back(temp_count);
		  temp_count=0;
		} else {
		  temp_count++;
		  if(temp_count==64) {
			act[x_dim][k].push_back(0);
			count[x_dim][k].push_back(temp_count);
		  	temp_count=0;
		  }
		}
	  }
	}
  }
}

void store_rle_in_file() {
  char x1[100] = "output_datasets/";
  char x2[100] = "output_datasets/";
  char x3[100] = "output_datasets/";
  char a[100] = "/act_val.data";
  char b[100] = "/act_index.data";
  char c[100] = "/act_ptr.data";

  // ofstream act_val ("output_datasets/act_val.data"); // to store nodes in rle
  ofstream act_val (strcat(strcat(x1,str.c_str()),a));
  // ofstream act_ind ("output_datasets/act_index.data"); // to store nodes in rle
  ofstream act_ind (strcat(strcat(x2,str.c_str()),b));

  if(act_val.is_open() && act_ind.is_open()) {
    int i=0, j=0;
    for(unsigned k=0; k<act[i][j].size(); ++k) {
      act_val << act[i][j][k] << "\n";
      act_ind << count[i][j][k] << "\n";
    }
  }
  act_val.close();
  act_ind.close();

  // ofstream act_ptr ("output_datasets/act_ptr.data");
  ofstream act_ptr (strcat(strcat(x3,str.c_str()),c));
  unsigned cum_size=0;
  int i=0, j=0;
  cum_size += act[i][j].size();
  act_ptr << cum_size << endl;
  act_ptr.close();
}
/*
void store_rle_in_file() {
  char x[100] = "output_datasets/";
  char a[100] = "/act_val.data";
  char b[100] = "/act_index.data";
  char c[100] = "/act_ptr.data";

  // ofstream act_val ("output_datasets/act_val.data"); // to store nodes in rle
  ofstream act_val (strcat(strcat(x,str.c_str()),a));
  // ofstream act_ind ("output_datasets/act_index.data"); // to store nodes in rle
  ofstream act_ind (strcat(strcat(x,str.c_str()),b));

  if(act_val.is_open() && act_ind.is_open()) {

	for(int i=0; i<dim1; i++) {
	  for(unsigned j=0; j<Ni; j++) {
		for(unsigned k=0; k<act[i][j].size(); ++k) {
		  act_val << act[i][j][k] << "\n";
		  act_ind << count[i][j][k] << "\n";
		}
	  }
    }
  }
  act_val.close();
  act_ind.close();

  // ofstream act_ptr ("output_datasets/act_ptr.data");
  ofstream act_ptr (strcat(strcat(x,str.c_str()),c));
  unsigned cum_size=0;
  for(int i=0; i<dim1; i++) {
    for(unsigned j=0; j<Ni; j++) {
	  cum_size += act[i][j].size();
	  act_ptr << cum_size << endl;
	}
  }
  act_ptr.close();
}
*/
int main() {
  str = net_name;
  FILE *act_file;
  // act_file = fopen("input_datasets/act_conv2.data", "r");
  // act_file = fopen("input_datasets/very_small/act.txt", "r");
  // act_file = fopen(str.c_str());
  char a[100] = "input_datasets/";
  char b[100] = "/act.txt";
  // act_file = fopen(strcat(str.c_str(),"act_conv.data"));
  act_file = fopen(strcat(strcat(a,str.c_str()),b), "r");
  char lineToRead[5000];

  int linear_index=0;
  int x, y, z;
  while(fgets(lineToRead, 5000, act_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
	float var;

	iss >> var;
	x = linear_index%Nx;
	y = (linear_index%(Nx*Ny))/Ny;
	z = linear_index/(Nx*Ny);
	dense_act[x][y][z] = var;
	linear_index++;
  }

  cout << "Done reading file!\n";

  encode_rle();
  store_rle_in_file();
  return 0;
}
