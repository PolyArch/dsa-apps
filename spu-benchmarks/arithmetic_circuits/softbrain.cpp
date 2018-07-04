#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "test.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>
#include <sstream>

struct tree {
  uint64_t* nodeType; 
  uint64_t* index;
  double* vr; 
  double* dr; 
  uint64_t* child0;
  uint64_t* child1;
  uint64_t* flag;
};

void backpropagation(tree circuit, int index, int nodes_at_level[d2-d1]){
  // 2 parents, overwrite the value? how is this done? but should not be the
  // problem in my example
  int n_times = 0;

  SB_CONFIG(test_config,test_size);

  int i=0;
  for (int d = 0; d < (d2-d1-1); d++) {
    // start with i
    n_times = nodes_at_level[d];
    // std::cout << "number of elements at the level: " << n_times << std::endl;
    SB_DMA_READ(&circuit.nodeType[i], 8, 8, n_times, P_test_nodeType);
    SB_DMA_READ(&circuit.dr[i], 8, 8, n_times, P_test_dr);
    SB_DMA_READ(&circuit.flag[i], 8, 8, n_times, P_test_flag);
    SB_DMA_READ(&circuit.vr[i], 8, 8, n_times, P_test_vr);
    SB_DMA_READ(&circuit.child0[i], 8, 8, n_times, P_IND_DOUB0);
    SB_DMA_READ(&circuit.child1[i], 8, 8, n_times, P_IND_1);
    SB_DMA_READ(&circuit.child1[i], 8, 8, n_times,P_IND_2);
    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT(P_IND_DOUB0, &circuit.vr[i], n_times, P_test_c1vr);
    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT(P_IND_1, &circuit.vr[i], n_times, P_test_c2vr);

    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT_WR(P_IND_DOUB1, &circuit.dr[i], n_times, P_test_c0dr);
    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT_WR(P_IND_2, &circuit.dr[i], n_times, P_test_c1dr);

    SB_WAIT_ALL();
    i+=nodes_at_level[d];
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
        arith_ckt.vr[index] = 1 + rand()/RAND_MAX; // What is uint? I want double, right?
        arith_ckt.dr[index] = 1 + rand()/RAND_MAX;
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
  begin_roi();
  backpropagation(arith_ckt, index, nodes_at_level);
  end_roi();
  sb_stats();
  printf("Backpropagation done!\n");
  return 0;
}
