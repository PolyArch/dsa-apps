#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <map>
#include <bitset>
#include <string.h>
#include <list>
#include <algorithm>
#include "assert.h"
using namespace std;

#define VEC_WIDTH 4

void read_input_file() {
  string str1(csr_file);
  ofstream csr_file1(str1.c_str());
 
  string str(data_file);
  FILE* graph_file = fopen(str.c_str(), "r");

  char linetoread[5000];

  // cout << "start reading graph input file!\n";
  int tile = V/NUM_THREADS;
  int count = 1;
  int cur_border = count*tile;
  --cur_border; // it has to be before next one starts

  while(fgets(linetoread, 5000, graph_file) != NULL) { // && csr_file1.is_open()) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    int src, dst, wgt;
    iss >> src >> dst >> wgt;  
    // cout << src << " " << dst << " " << wgt << endl;
    // Instead of padding
    if(dst>=V) dst = V-1;
    if(src >= V) continue;
    // if(src>=cur_border) { // should be after 1676
    if(src>cur_border) { // should be after 1676
      cout << "Current border is: " << cur_border << endl;
      cout << "IDENTIFIED BORDER, putting pad\n";
      // csr_file1 << "border" << endl;
      for(int i=0; i<2*VEC_WIDTH; ++i) {
        csr_file1 << "0 1" << endl;
      }
      cur_border = (++count)*tile;
      --cur_border;
    }
    csr_file1 << src << " " << dst << endl;
  }
  for(int i=0; i<2*VEC_WIDTH; ++i) {
    csr_file1 << "0 1" << endl;
  }

  fclose(graph_file);
  cout << "Done reading graph file!\n";
}

int main() {
  read_input_file();
  return 0;
}