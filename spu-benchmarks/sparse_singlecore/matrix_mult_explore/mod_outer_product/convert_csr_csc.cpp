#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <assert.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <map>
#include <inttypes.h>

using namespace std;

#define N 4
#define M 4096

uint16_t matrix[N][M];

int main() {
  char lineToRead[5000];

  FILE *csr_file = fopen("input_weights.data", "r");

  printf("Start reading csr filE\n");
  
  while(fgets(lineToRead, 5000, csr_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    int first_index, second_index, val;

	iss >> first_index >> second_index >> val;
    matrix[first_index][second_index] = val;
  }  
  fclose(csr_file);

  printf("Done reading csr filE\n");
  
  printf("Start writing csc filE\n");

  ofstream csc_file ("input_csc_weights.data");

  if(csc_file.is_open()) {
    for(int i=0; i<M; ++i) {
      for(int j=0; j<N; ++j) {
        if(matrix[j][i]!=0) {
          csc_file << j << " " << i << " " << matrix[j][i] << endl;
        }
      }
    }
  }

  csc_file.close(); 
  printf("Done writing csc filE\n");

  return 0;
}
