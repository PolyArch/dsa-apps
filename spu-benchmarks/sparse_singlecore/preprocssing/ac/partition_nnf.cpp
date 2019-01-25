/****
Input: circuit.data and height.data (DAG in topological level-order)
Step2: Partition the graph according to the number of nodes
Step1: Insert shadow nodes in the graph (at the partition layer)
Output: new circuit.data and new height.data (should need to copy h2-h1 nodes only; also inc height at intermediate node) and part.data
***/
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

#define MAX_SIZE 1000
#define N_PART 64

// I still think it should be an struct of arrays (we cannot read these things
// in a single port -- stride will affect performance) -- can copy it there
struct prop {
 char nodeType;
 float vr;
 float dr;
 bool flag;
};

// this is we want to read from file
struct ac_node {
  // struct prop node_prop;
  char nodeType;
  float vr;
  int c0; int c1;
};

vector<int> copy_nodes;
vector<int> height_ptr; // height ptr for copy nodes
int part_size;

// input DAG properties
int V; // total number of vertices
vector<struct ac_node> ac; // length should be V
vector<int> orig_hgt_ptr; // original height pointers

vector<int> final_ac_ind; // length should be V
vector<int> final_hgt_ptr; // original height pointers

void find_copy_nodes() {
  // go over all the vertices from last partition
  height_ptr.push_back(0);
  for(int i=part_size; i<V; i+=part_size){
	// height_ptr.push_back(copy_nodes.size());
	for(int j=i; j<(i+part_size) && j<V; j++){
	  struct ac_node temp = ac[j];
	  if(temp.c0>-1 && temp.c0 < i){
		copy_nodes.push_back(temp.c0);
		temp.c0 = indexOf(copy_nodes[copy_nodes.size()-1]);
	  }
	  if(temp.c1>-1 && temp.c1 < i){
		copy_nodes.push_back(temp.c1);
		temp.c1 = indexOf(copy_nodes[copy_nodes.size()-1]);
	  }
	}
	height_ptr.push_back(copy_nodes.size());
	// cout << "size: " << copy_nodes.size() << "\n";
  }
}

// load balance (compute balance) while minimizing number of copy nodes (memory balance)
// this is just the load balance
void insert_copy_nodes(){
  int cur_hgt=0;
  int part_id = 0;

  for(int i=0; i<V; i+=part_size){
    part_id = i/part_size;
	for(int j=i; j<(i+part_size) && j<V; ++j){
	  // first copy all the required nodes
	  final_ac_ind.push_back(j);
	  // fix the child nodes of node j here
	  // struct ac_node temp=ac[j];
	  // if it is in same partition



	  if(j==orig_hgt_ptr[cur_hgt]) { // if it was same height
		final_hgt_ptr.push_back(final_ac_ind.size());
		cur_hgt++;
	  }
	}
	// TODO: could save space for copy nodes (also add erase)
	
	// now the copy nodes (no need after the last one)
	if(i<V-part_size) {
	  int copy_node_size = height_ptr[part_id+1]-height_ptr[part_id];
	  cout << "copy nodes size at this partition id: " << part_id << " is: " << copy_node_size << "\n";
	  // for(int k=0; k<height_ptr[part_id]; ++k){
	  for(int k=height_ptr[part_id]; k<height_ptr[part_id+1]; ++k){
	    final_ac_ind.push_back(copy_nodes[k]);
	  }
	  // how to say that there is a partition here (I need to change the
	  // heights)
	  final_hgt_ptr.push_back(final_ac_ind.size());
	}
  }
}

// TODO: how do I know which nodes are copy nodes (some validity information)
void store_ac_in_file() {
  ofstream ac_file ("final_circuit.data"); // to store the nodes in top_order

  // write to the file here
  if(ac_file.is_open()) {
    ac_file << "nnf " << final_ac_ind.size() << "\n";
    for(unsigned i=0; i<final_ac_ind.size(); i++) {
	  /*
	  if(final_ac_ind[i]>(V-1)) {
		cout << "index is greater than the index in arithmetic circuit\n";
	  }
      if(final_ac_ind[i]<0) {
		cout << "index is less than 0\n";
	  }
	  */
	  // this is not direct mapping as earlier
      struct ac_node temp = ac[final_ac_ind[i]];
      if(temp.nodeType=='l') {
         ac_file << temp.nodeType << " " << temp.vr << "\n";
      } else {
		// TODO: should be either real child (if in same partition) or copy node (if
		// in earlier partition)
        ac_file << temp.nodeType << " " << temp.c0 << " " << temp.c1 << "\n";
      }
    }
  }
  ac_file.close();
  cout << "Done writing circuit\n";

  ofstream height_index("final_index.data"); // to store the nodes in top_order

  if(height_index.is_open()) {
    for(unsigned i=0; i<final_hgt_ptr.size(); i++) {
	  height_index << final_hgt_ptr[i] << "\n";
    }
  }
  height_index.close();
  cout << "Done writing index\n";
}


int main() {
  FILE *ac_file;
  ac_file = fopen("circuit.data", "r");
  // ac_file = fopen("pigs.uai.ac", "r");
  char lineToRead[MAX_SIZE];
  int cur_v=0;

  while (fgets(lineToRead, MAX_SIZE, ac_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    char op;
    int ignore;
    float var;
	int child0, child1;

	iss >> op;
    if(op=='n') {
      iss >> level >> V;
	  part_size = V/N_PART;
      ac.resize(V);
      continue;
    }

    struct ac_node temp;
    temp.nodeType = op;
    if(op=='l') {
      iss >> var; // just vr?
      temp.c0 = -1; temp.c1 = -1;
      temp.vr = var;
    } else {
	  iss >> ignore >> child0 >> child1; //ignore because we are assuming binary trees
	  temp.c0 = child0; temp.c1 = child1;
      temp.vr = var;
	}

	ac[cur_v] = temp;
	cur_v++;
  }

  fclose(ac_file);

  FILE *height_file;
  height_file = fopen("index.data", "r");
  int x;

  while (fgets(lineToRead, MAX_SIZE, height_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
	iss >> x;
	orig_hgt_ptr.push_back(x);
  }
  fclose(height_file);
  cout << "Done reading inputs ACs\n";

  find_copy_nodes();
  cout << "Stored copy nodes in variable\n";
  insert_copy_nodes();
  cout << "Inserted into final AC\n";
  store_ac_in_file();
  cout << "Final store done\n";

  return 0;
}
