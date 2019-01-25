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
  vector<int> out_nodes;
  int c0; int c1;
};

vector<int> top_order;
vector<int> height_ptr; // length=height of the DAG

// input DAG properties
int V; // total number of vertices
vector<struct ac_node> ac; // length should be V

vector<int> in_degree; // set during reading file (number of children of each node)
queue<int> q;

void kahn_algo(){

  int cur_q_ptr = q.size(); // to note the height of that node
  int cur_res_ptr = 0;
  height_ptr.push_back(cur_res_ptr);

  // cout << "topological order: ";
  while(!q.empty()) {
    int u = q.front(); // index of that vertex
    q.pop();
    cur_q_ptr--;

    top_order.push_back(u);
	  // cout << u << " ";
    bool check = (cur_q_ptr==0); // FIXME: should be equal to the ptr?

    // for all the out nodes
    for(unsigned i=0; i<ac[u].out_nodes.size(); ++i) {
  	  if(--in_degree[ac[u].out_nodes[i]]==0){
  	    q.push(ac[u].out_nodes[i]);
  	  }
    }

    if(check){
  	  cur_q_ptr = q.size(); // mark it at the end
  	  cur_res_ptr = top_order.size();
	    height_ptr.push_back(cur_res_ptr);
    }
  }
  // cout << "\n";
  cout << "Total size: " << top_order.size() << " with vertices: " << V << endl;
}

void store_bfs_in_file() {
  ofstream ac_file ("circuit.data"); // to store the nodes in top_order

  // write to the file here
  if(ac_file.is_open()) {
    ac_file << "nnf " << V << "\n";
    for(int i=0; i<V; i++) {
      struct ac_node temp = ac[top_order[i]];
      if(temp.nodeType=='l') {
         ac_file << temp.nodeType << " " << temp.vr << "\n";
      } else {
        ac_file << temp.nodeType << " " << top_order[temp.c0] << " " << top_order[temp.c1] << "\n";
      }
    }
  }
  ac_file.close();

  ofstream height_index("index.data"); // to store the nodes in top_order

  if(height_index.is_open()) {
    for(unsigned i=0; i<height_ptr.size(); i++) {
	  height_index << height_ptr[i] << "\n";
    }
  }
  height_index.close();
}

int main() {
  FILE *ac_file;
  ac_file = fopen("water.uai.ac", "r");
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
    // V = 45456;
    ac.resize(V);
    continue;
  }

	struct ac_node temp;
  temp.nodeType = op;
	if(op=='l') {
	  iss >> var; // just vr?
	  temp.c0 = -1; temp.c1 = -1;
    temp.vr = var;
	  in_degree.push_back(0);
	  q.push(cur_v);

	} else {
	  iss >> ignore >> child0 >> child1; //ignore because we are assuming binary trees
	  temp.c0 = child0; temp.c1 = child1;
    temp.vr = var;
	  in_degree.push_back(2); // this should ideally be ignore
	  // set its' parents
	  ac[temp.c0].out_nodes.push_back(cur_v);
	  ac[temp.c1].out_nodes.push_back(cur_v);
	}

	ac[cur_v] = temp;
	cur_v++;

  }

  fclose(ac_file);
  cout << "Done reading the file\n";

  kahn_algo();
  cout << "Done topological sorting\n";
  store_bfs_in_file();
  cout << "Done writing the file\n";

  return 0;
}
