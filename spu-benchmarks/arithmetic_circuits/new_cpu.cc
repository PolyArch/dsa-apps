#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <stdbool.h>
#include <time.h>
#include <string>
#include <sstream>
#include <inttypes.h>

#define d1 3
#define d2 6

struct tree {
  uint64_t* nodeType; 
  uint64_t* index;
  double* vr; 
  double* dr; 
  uint64_t* child0;
  uint64_t* child1;
  uint64_t* flag;
};

void backpropagation(tree circuit, int index, int cum_nodes_at_level[d2-d1]){
  // pid is parent id
  int c0_id, c1_id;
  // index is the leaf node here? should start from root
  for (int i = 0; i < index; i+=cum_nodes_at_level[i]) {
    for(int pid = i; pid < cum_nodes_at_level[i+1]; pid++) {
      c0_id = circuit.child0[pid];
      c1_id = circuit.child1[pid];

      if (circuit.nodeType[pid] == 0) {
        circuit.dr[c0_id] = circuit.dr[pid];
        circuit.dr[c1_id] = circuit.dr[pid];
      }
      else if (circuit.nodeType[pid] == 1) {
        if (circuit.dr[pid] == 0) {
          circuit.dr[c0_id] = 0;
          circuit.dr[c1_id] = 0;
        } else if (circuit.flag[pid]) {
	      if (circuit.vr[c0_id] == 0) {
            circuit.dr[c0_id] = circuit.dr[pid] * circuit.vr[pid];
            circuit.dr[c1_id] = 0;
	      } else {
	        circuit.dr[c0_id] = circuit.dr[pid] * (circuit.vr[pid] / circuit.vr[c0_id]);
            circuit.dr[c1_id] = 0;
	      }
        } else {
          circuit.dr[c0_id] = circuit.dr[pid] * (circuit.vr[pid] / circuit.vr[c0_id]);
          circuit.dr[c1_id] = circuit.dr[pid] * (circuit.vr[pid] / circuit.vr[c1_id]);
        }
      }
    }
  }
}


int main() {
  FILE *ac_file;
  char lineToRead[50000]; 
  int nodes_at_level[d2-d1];
  int cum_nodes_at_level[d2-d1];
  int cur_level=0;
  struct tree arith_ckt;
  int index = 0;
  int *a, *b, *c, *d;

  // circuit = (struct node**)malloc(sizeof(struct node*) * 50000);
  arith_ckt.nodeType = (uint64_t*)malloc(sizeof(uint64_t) * 50000);
  arith_ckt.index = (uint64_t*)malloc(sizeof(uint64_t) * 50000);
  arith_ckt.vr = (double*)malloc(sizeof(double) * 50000);
  arith_ckt.dr = (double*)malloc(sizeof(double) * 50000);
  arith_ckt.child0 = (uint64_t*)malloc(sizeof(uint64_t) * 50000);
  arith_ckt.child1 = (uint64_t*)malloc(sizeof(uint64_t) * 50000);
  arith_ckt.flag = (uint64_t*)malloc(sizeof(uint64_t) * 50000);
  a = (int*)malloc(sizeof(int) * 50000);
  b = (int*)malloc(sizeof(int) * 50000);
  c = (int*)malloc(sizeof(int) * 50000);
  d = (int*)malloc(sizeof(int) * 50000);
  
  ac_file = fopen("input.data", "r");

  printf("Started reading file!\n");

  while (fgets(lineToRead, 50000, ac_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    char op;
    int n;
    iss >> level >> n >> op;
    
    nodes_at_level[cur_level] = n;
    if(cur_level==0){
      cum_nodes_at_level[cur_level] = n;
    } else {
      cum_nodes_at_level[cur_level] = (n+cum_nodes_at_level[cur_level-1]);
    }
    char ignore;
    for (int i = 0; i < n * 2; i+=2) {
        iss >> a[index] >> ignore >> b[index] >> ignore >> c[index] >> ignore >> d[index];
        if(op=='*'){
          arith_ckt.nodeType[index]=1;
        } else {
          arith_ckt.nodeType[index]=0;
        }
        arith_ckt.vr[index] = rand();
        arith_ckt.dr[index] = rand();
        arith_ckt.flag[index] = rand()%2;
        index++;
    }
    cur_level++;
  }

  int child1_ind, child2_ind;
  for(int i=0; i<index; i++){
    child1_ind = cum_nodes_at_level[a[i]-1]+b[i];
    child2_ind = cum_nodes_at_level[c[i]-1]+d[i];
    arith_ckt.child0[i] = child1_ind;
    arith_ckt.child1[i] = child2_ind;
  }
  
  printf("Done reading file!\n");

  printf("Starting backpropagation with number of nodes: %d\n", index);
  // begin_roi();
  backpropagation(arith_ckt, index, cum_nodes_at_level);
  // end_roi();
  printf("Backpropagation done!\n");
  return 0;
}
