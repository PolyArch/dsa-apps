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
#include <math.h>

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

// copy nodes to be accessed by this partition (for 0, it's empty)
vector<int> copy_nodes[N_PART];
// vector<int> height_ptr; // height ptr for copy nodes
int part_size;

// input DAG properties
int V; // total number of vertices
vector<struct ac_node> ac; // length should be V
vector<int> orig_hgt_ptr; // original height pointers

vector<int> final_ac_ind; // length should be V
vector<int> final_hgt_ptr; // original height pointers

int cumsum(int x){
  if(x<=0)	return 0;
  int s=0;
  for(int i=1; i<x; ++i){
	s+= copy_nodes[i].size();
  }
  return s;
}

// Should be either real child (if in same partition) or copy node (if
// in earlier partition)
// TODO: confirm the formulas
void find_copy_nodes() {
  // go over all the vertices from last partition
  int part_id = 0;
  int x = 0;

  for(int i=part_size; i<V; i+=part_size){
	part_id = i/part_size;
	for(int j=i; j<(i+part_size) && j<V; j++){
	  struct ac_node temp = ac[j];
	  if(temp.c0>-1 && temp.c0 < i){
		copy_nodes[part_id].push_back(temp.c0);
		// ac[j].c0 = j + cumsum(part_id-1) + copy_nodes[part_id].size()-1;
	    x = part_size*part_id + cumsum(part_id-1) + copy_nodes[part_id].size()-1; // copy nodes would be put before this
	  } else { // if in the same partition
		x = temp.c0 + part_size*part_id + cumsum(part_id-1);
	  }
	  ac[j].c0 = x;

	  if(temp.c1>-1 && temp.c1 < i){
		copy_nodes[part_id].push_back(temp.c1);
		ac[j].c1 = j + cumsum(part_id-1) + copy_nodes[part_id].size()-1;
		// temp.c1 = j + cumsum(part_id-1) + copy_nodes[part_id].size()-1;
		// x = j + cumsum(part_id-1) + copy_nodes[part_id].size()-1;
	    x = part_size*part_id + cumsum(part_id-1) + copy_nodes[part_id].size()-1; // copy nodes would be put before this
	    ac[j].c1 = x;
	  } else {
		x = temp.c1 + part_size*part_id + cumsum(part_id-1);
	  }
	  ac[j].c1 = x;
	}
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
	  if(j==orig_hgt_ptr[cur_hgt]) { // if it was same height
		final_hgt_ptr.push_back(final_ac_ind.size());
		cur_hgt++;
	  }
	}
	// TODO: could save space for copy nodes (also add erase)
	// now the copy nodes (no need after the last one)
	if(i<V-part_size) {
	  // cout << "copy nodes size at this partition id: " << part_id << " is: " << copy_nodes[part_id+1].size() << "\n";
	  for(unsigned k=0; k<copy_nodes[part_id+1].size(); ++k){
	    final_ac_ind.push_back(copy_nodes[part_id+1][k]);
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
	  // this is not direct mapping as earlier
      struct ac_node temp = ac[final_ac_ind[i]];
      if(temp.nodeType=='l') {
         ac_file << temp.nodeType << " " << temp.vr << "\n";
      } else {
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
    float var;
	int child0, child1;

	iss >> op;
    if(op=='n') {
      iss >> level >> V;
	  part_size = ceil(V/float(N_PART));
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
	  iss >> child0 >> child1; //ignore because we are assuming binary trees
	  temp.c0 = child0; temp.c1 = child1;
      temp.vr = 0;
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
